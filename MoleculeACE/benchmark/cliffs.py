"""
Author: Derek van Tilborg -- TU/e -- 25-05-2022

Code to compute activity cliffs

    - ActivityCliffs:                   Class that takes care of computing activity cliffs
    - find_fc():                        Calculate the fold change
    - get_fc():                         Compute the pairwise fold change
    - get_levenshtein_matrix():         Compute the pairwise Levenshtein similarity
    - get_tanimoto_matrix():            Compute the pairwise Tanimoto similarity
    - get_scaffold_matrix():            Compute the pairwise scaffold similarity
    - get_mmp_matrix():                 Compute a matrix of Matched Molecular Pairs
    - mmp_similarity():                 Compute binary mmp similarity matrix
    - moleculeace_similarity():         Compute the consensus similarity (being >0.9 in at least one similarity type)

"""

from typing import List, Callable, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdMMPA import FragmentMol
from Levenshtein import distance as levenshtein
from tqdm import tqdm

from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol


class ActivityCliffs:
    """ Activity cliff class that computes cliff compounds """
    def __init__(self, smiles: List[str], bioactivity: Union[List[float], np.array]):
        self.smiles = smiles
        self.bioactivity = list(bioactivity) if type(bioactivity) is not list else bioactivity
        self.cliffs = None

    def find_cliffs(self, similarity: float = 0.9, potency_fold: float = 10, mmp: bool = False,
                    custom_cliff_function: Callable = None):
        """ Compute activity cliffs

        :param similarity: (float) threshold value to determine structural similarity
        :param potency_fold: (float) threshold value to determine difference in bioactivity
        :param custom_cliff_function: (Callable) function that takes: smiles: List[str] and similarity: float and
          returns a square binary matrix where 1 is similar and 0 is not.
        :param mmp: (bool) use matched molecular pairs to determine similarity instead
        """

        if mmp:
            sim = mmp_similarity(self.smiles)
        else:
            sim = moleculeace_similarity(self.smiles, similarity)

        if custom_cliff_function is not None:
            custom_sim = custom_cliff_function(self.smiles, similarity)
            sim = np.logical_or(sim == 1, custom_sim == 1).astype(int)

        fc = (get_fc(self.bioactivity) > potency_fold).astype(int)

        self.cliffs = np.logical_and(sim == 1, fc == 1).astype(int)

        return self.cliffs

    def get_cliff_molecules(self, return_smiles: bool = True, **kwargs):
        """

        :param return_smiles: (bool) return activity cliff molecules as a list of SMILES strings
        :param kwargs: arguments for ActivityCliffs.find_cliffs()
        :return: (List[int]) returns a binary list where 1 means activity cliff compounds
        """
        if self.cliffs is None:
            self.find_cliffs(**kwargs)

        if return_smiles:
            return [self.smiles[i] for i in np.where((sum(self.cliffs) > 0).astype(int))[0]]
        else:
            return list((sum(self.cliffs) > 0).astype(int))

    def __repr__(self):
        return "Activity cliffs"


def find_fc(a: float, b: float):
    """Get the fold change of to bioactivities"""

    return max([a, b]) / min([a, b])


def get_fc(bioactivity: List[float]):
    """ Calculates the pairwise fold difference in compound activity given a list of activities"""

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


def get_levenshtein_matrix(smiles: List[str], normalize: bool = True, hide: bool = False, top_n: int = None):
    """ Calculates a matrix of levenshtein similarity scores for a list of SMILES string"""

    smi_len = len(smiles)

    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    for i in tqdm(range(smi_len if top_n is None else top_n), disable=hide):
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


def get_tanimoto_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024, hide: bool = False, top_n: int = None):
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
    for i in tqdm(range(smi_len if top_n is None else top_n), disable=hide):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_fp[smiles[i]],
                                                     db_fp[smiles[j]])
    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_scaffold_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024, hide: bool = False, top_n: int = None):
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
    for i in tqdm(range(smi_len if top_n is None else top_n), disable=hide):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_scaf[smiles[i]],
                                                     db_scaf[smiles[j]])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def find_fragments(smiles: List[str]):
    """ Build a database of molecular fragments for matched molecular pair analysis. Molecular fragmentation from the
    Hussain and Rae algorithm is used. We only use a single cut (true to the original MMP idea) """

    db = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        cuts = FragmentMol(m, maxCuts=1, resultsAsMols=False)

        # extract all not None fragments into a flat list
        fragments = sum([[i for i in cut if i != ''] for cut in cuts], [])

        # Keep the largest fragment as the core structure.
        for i, frag in enumerate(fragments):
            split_frags = frag.split('.')
            if Chem.MolFromSmiles(split_frags[0]).GetNumAtoms() >= Chem.MolFromSmiles(split_frags[-1]).GetNumAtoms():
                core = split_frags[0]
            else:
                core = split_frags[-1]

            # Ignore dummy variables for matching (otherwise you will never get a match). Dummies are introduced during
            # fragmentation
            qp = Chem.AdjustQueryParameters()
            qp.makeDummiesQueries = True
            qp.adjustDegreeFlags = Chem.ADJUST_IGNOREDUMMIES
            fragments[i] = Chem.AdjustQueryProperties(Chem.MolFromSmiles(core), qp)
        # Add all core fragments to the dictionary
        db[smi] = fragments

    return db


def mmp_match(smiles: str, fragments: List):
    """ Check if fragments (provided as rdkit Mol objects are substructures of another molecule (from SMILES string) """

    m = Chem.MolFromSmiles(smiles)
    for frag in fragments:
        # Match fragment on molecule
        if m.HasSubstructMatch(frag):
            return 1
    return 0


def get_mmp_matrix(smiles: List[str], hide: bool = False):
    """ Calculates a matrix of matched molecular pairs for a list of SMILES string"""

    # Make a fingerprint database
    db_frags = find_fragments(smiles)

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix.
    for i in tqdm(range(smi_len), disable=hide):
        for j in range(i, smi_len):
            m[i, j] = mmp_match(smiles[i], db_frags[smiles[j]])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))

    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def mmp_similarity(smiles: List[str], similarity=None):
    """ Calculate which pairs of molecules are matched molecular pairs """

    return (get_mmp_matrix(smiles) > 0).astype(int)


def moleculeace_similarity(smiles: List[str], similarity: float = 0.9, hide: bool = False):
    """ Calculate which pairs of molecules have a high tanimoto, scaffold, or SMILES similarity """

    m_tani = get_tanimoto_matrix(smiles, hide=hide) >= similarity
    m_scaff = get_scaffold_matrix(smiles, hide=hide) >= similarity
    m_leve = get_levenshtein_matrix(smiles, hide=hide) >= similarity

    return (m_tani + m_scaff + m_leve).astype(int)


def is_cliff(smiles1, smiles2, y1, y2, similarity: float = 0.9, potency_fold: float = 10):
    """ Calculates if two molecules are activity cliffs """
    sim = moleculeace_similarity([smiles1, smiles2], similarity=similarity, hide=True)[0][1]
    fc = get_fc([y1, y2])[0][1]

    return sim == 1 and fc >= potency_fold
