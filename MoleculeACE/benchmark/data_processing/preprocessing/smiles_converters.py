"""
Code to encode descriptors from SMILES strings
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from MoleculeACE.benchmark import data_processing
from .one_hot_encoding import is_acceptable_smiles, OneHotEncode, smiles_encoding


def smiles_to_morgan2(smiles, radius: int = 2, nbits: int = 1024):
    """Calculate the morgan fingerprint"""
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    fp = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) for m in mols]
    return rdkit_numpy_convert(fp)


def smiles_to_rdkit(smiles: list):
    """Calculate the rdkit fingerprint"""
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    fp = [Chem.RDKFingerprint(m) for m in mols]
    return rdkit_numpy_convert(fp)


def smiles_to_atom_pair(smiles: list):
    """Calculate the atom-pair fingerprint"""
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    fp = [AllChem.GetHashedAtomPairFingerprintAsBitVect(m) for m in mols]
    return rdkit_numpy_convert(fp)


def smiles_to_topological_torsion(smiles: list):
    """Calculate the topological fingerprint fingerprint"""
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    fp = [AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(m) for m in mols]
    return rdkit_numpy_convert(fp)


def smiles_to_smarts(smiles: list):
    """Calculate the topological fingerprint fingerprint"""
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    fp = [Chem.PatternFingerprint(m) for m in mols]
    return rdkit_numpy_convert(fp)


def smiles_to_maccs(smiles: list):
    """Calcualte MACCS fingerprint"""
    from rdkit.Chem import MACCSkeys
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    fp = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    return rdkit_numpy_convert(fp)


def smiles_to_drug_like(smiles: list):
    """Calcualte drug-like fingerprint"""
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    return np.asarray([data_processing.drug_like_descriptor(m) for m in mols])


def smiles_to_whim(smiles: list):
    """Calculate WHIM descriptors, remember failed molecules"""
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    failed_mols = []
    whims = []
    for idx, m in enumerate(mols):
        try:
            whims.append(np.array(data_processing.whim_descriptor(m)))
        except:
            failed_mols.append(idx)
    if len(failed_mols) > 0:
        print(
            f"Could not calculate WHIM descriptors for {len(failed_mols)} mols because no conformers could be found.")

    return np.asarray(whims), failed_mols


def smiles_to_canonical_graph(smiles: list):
    """Calculate canonical dgl graphs for smiles, remember failed molecules"""
    from dgllife.utils import smiles_to_bigraph
    from dgllife.utils import CanonicalBondFeaturizer
    from dgllife.utils import CanonicalAtomFeaturizer

    graphs = []
    failed_mols = []
    for idx, smi in enumerate(smiles):
        try:
            graph = smiles_to_bigraph(smi, add_self_loop=True,
                                      node_featurizer=CanonicalAtomFeaturizer(),
                                      edge_featurizer=CanonicalBondFeaturizer(self_loop=True))
            graphs.append(graph)
        except:
            failed_mols.append(idx)

    return graphs, failed_mols


def smiles_to_attentive_graph(smiles: list):
    """Calculate attentivefp dgl graphs for smiles, remember failed molecules"""
    from dgllife.utils import smiles_to_bigraph
    from dgllife.utils import AttentiveFPAtomFeaturizer
    from dgllife.utils import AttentiveFPBondFeaturizer

    graphs = []
    failed_mols = []
    for idx, smi in enumerate(smiles):
        try:
            graph = smiles_to_bigraph(smi, add_self_loop=True,
                                      node_featurizer=AttentiveFPAtomFeaturizer(),
                                      edge_featurizer=AttentiveFPBondFeaturizer(self_loop=True))
            graphs.append(graph)
        except:
            failed_mols.append(idx)

    return graphs, failed_mols


def rdkit_numpy_convert(fp):
    output = []
    for f in fp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)


def smiles_to_onehot(smiles: list):
    """ Return a list of smiles, remember failed molecules"""
    failed_molecules = []
    acceptable_smiles = []

    for idc, smi in enumerate(smiles):
        if is_acceptable_smiles(smi):
            acceptable_smiles.append(smi)
        else:
            failed_molecules.append(idc)


    OneHotEncoder = OneHotEncode(max_len_model=smiles_encoding['max_smiles_len'] + 2,
                                 n_chars=smiles_encoding['vocab_size'],
                                 indices_token=smiles_encoding['indices_token'],
                                 token_indices=smiles_encoding['token_indices'],
                                 pad_char=smiles_encoding['pad_char'],
                                 start_char=smiles_encoding['start_char'],
                                 end_char=smiles_encoding['end_char'])

    # one_hot_smi = OneHotEncoder.generator_smile_to_onehot(smi)
    onehots = OneHotEncoder.smile_to_onehot(acceptable_smiles)

    return onehots, failed_molecules