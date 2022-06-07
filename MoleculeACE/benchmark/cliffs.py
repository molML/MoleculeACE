"""
Author: Derek van Tilborg -- TU/e -- 25-05-2022

Code to compute activity cliffs

"""


from typing import List, Callable, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from Levenshtein import distance as levenshtein
from tqdm import tqdm

from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol


class ActivityCliffs:
    def __init__(self, smiles: List[str], bioactivity: Union[List[float], np.array]):
        self.smiles = smiles
        self.bioactivity = list(bioactivity) if type(bioactivity) is not list else bioactivity
        self.cliffs = None

    def find_cliffs(self, similarity: float = 0.9, potency_fold: float = 10, in_log10: bool = True,
                    custom_cliff_function: Callable = None):

        sim = moleculeace_similarity(self.smiles, similarity)

        if custom_cliff_function is not None:
            custom_sim = custom_cliff_function(self.smiles, similarity)
            sim = np.logical_or(sim == 1, custom_sim == 1).astype(int)

        fc = (get_fc(self.bioactivity, in_log10=in_log10) > potency_fold).astype(int)

        self.cliffs = np.logical_and(sim == 1, fc == 1).astype(int)

        return self.cliffs

    def get_cliff_molecules(self, return_smiles: bool = True, **kwargs):
        if self.cliffs is None:
            self.find_cliffs(**kwargs)

        if return_smiles:
            return [self.smiles[i] for i in np.where((sum(self.cliffs) > 0).astype(int))[0]]
        else:
            return list((sum(self.cliffs) > 0).astype(int))

    def __repr__(self):
        return "Activity cliffs"


def find_fc(a: float, b: float):
    """Get the fold change of to bioactivities (deconvert from log10 if needed)"""
    return max([a, b]) / min([a, b])


def get_fc(bioactivity: List[float], in_log10: bool = True):
    """ Calculates the pairwise fold difference in compound activity given a list of activities"""

    bioactivity = 10 ** abs(np.array(bioactivity)) if in_log10 else bioactivity

    act_len = len(bioactivity)
    m = np.zeros([act_len, act_len])
    # Calculate upper triangle of matrix
    for i in range(act_len):
        for j in range(i, act_len):
            m[i, j] = find_fc(bioactivity[i], bioactivity[j])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_levenshtein_matrix(smiles: List[str], normalize: bool = True):
    """ Calculates a matrix of levenshtein similarity scores for a list of SMILES string"""

    smi_len = len(smiles)

    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    for i in tqdm(range(smi_len)):
        for j in range(i, smi_len):
            if normalize:
                m[i, j] = levenshtein(smiles[i], smiles[j]) / max(len(smiles[i]), len(smiles[j]))
            else:
                m[i, j] = levenshtein(smiles[i], smiles[j])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Get from a distance to a similarity
    m = 1 - m

    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_tanimoto_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024):
    """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string"""

    # Make a fingerprint database
    db_fp = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        db_fp[smi] = fp

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    for i in tqdm(range(smi_len)):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_fp[smiles[i]],
                                                     db_fp[smiles[j]])
    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_scaffold_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024):
    """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string """

    # Make scaffold database
    db_scaf = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        try:
            skeleton = GraphFramework(m)
        except Exception:  # In the very rare case this doesn't work, use a normal scaffold
            print(f"Could not create a generic scaffold of {smi}, used a normal scaffold instead")
            skeleton = GetScaffoldForMol(m)
        skeleton_fp = AllChem.GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)
        db_scaf[smi] = skeleton_fp

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    for i in tqdm(range(smi_len)):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_scaf[smiles[i]],
                                                     db_scaf[smiles[j]])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def moleculeace_similarity(smiles: List[str], similarity: float = 0.9):
    """ Calculate which pairs of molecules have a high tanimoto, scaffold, or SMILES similarity """

    m_tani = get_tanimoto_matrix(smiles) >= similarity
    m_scaff = get_scaffold_matrix(smiles) >= similarity
    m_leve = get_levenshtein_matrix(smiles) >= similarity

    return (m_tani + m_scaff + m_leve).astype(int)
