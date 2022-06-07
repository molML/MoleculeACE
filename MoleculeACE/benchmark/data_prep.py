"""
Author: Derek van Tilborg -- TU/e -- 22-05-2022

A collection of data-prepping functions
    - split_data():             split ChEMBL csv into train/test taking similarity and cliffs into account
    - load_data():              load a pre-processed dataset from the benchmark
    - fetch_data():             download molecular bioactivity data from ChEMBL for a specific drug target

"""
from MoleculeACE.benchmark.cliffs import ActivityCliffs, get_tanimoto_matrix
from MoleculeACE.benchmark.const import RANDOM_SEED
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from rdkit import Chem
from typing import List
import pandas as pd
import numpy as np
import random
from tqdm import tqdm


def split_data(smiles: List[str], bioactivity: List[float], in_log10: bool = False, n_clusters: int = 5,
               test_size: float = 0.2, similarity: float = 0.9, potency_fold: int = 10, remove_stereo: bool = False):

    if remove_stereo:
        stereo_smiles_idx = [smiles.index(i) for i in find_stereochemical_siblings(smiles)]
        smiles = [smi for i, smi in enumerate(smiles) if i not in stereo_smiles_idx]
        bioactivity = [act for i, act in enumerate(bioactivity) if i not in stereo_smiles_idx]
        if len(stereo_smiles_idx) > 0:
            print(f"Removed {len(stereo_smiles_idx)} stereoisomers")

    if not in_log10:
        # bioactivity = (10**abs(np.array(bioactivity))).tolist()
        bioactivity = (-np.log10(bioactivity)).tolist()

    cliffs = ActivityCliffs(smiles, bioactivity)
    cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, similarity=similarity, potency_fold=potency_fold)

    # Perform spectral clustering on a tanimoto distance matrix
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=RANDOM_SEED, affinity='precomputed')
    clusters = spectral.fit(get_tanimoto_matrix(smiles)).labels_

    train_idx, test_idx = [], []
    for cluster in range(n_clusters):

        cluster_idx = np.where(clusters == cluster)[0]
        clust_cliff_mols = [cliff_mols[i] for i in cluster_idx]

        # Can only split stratiefied on cliffs if there are at least 2 cliffs present, else do it randomly
        if sum(clust_cliff_mols) > 2:
            clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=test_size,
                                                               random_state=RANDOM_SEED,
                                                               stratify=clust_cliff_mols, shuffle=True)
        else:
            clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=test_size,
                                                               random_state=RANDOM_SEED,
                                                               shuffle=True)

        train_idx.extend(clust_train_idx)
        test_idx.extend(clust_test_idx)

    train_test = []
    for i in range(len(smiles)):
        if i in train_idx:
            train_test.append('train')
        elif i in test_idx:
            train_test.append('test')
        else:
            raise ValueError(f"Can't find molecule {i} in train or test")

    return pd.DataFrame({'smiles': smiles,
                         'exp_mean [nM]': (10**abs(np.array(bioactivity))).tolist(),
                         'y': bioactivity,
                         'cliff_mol': cliff_mols,
                         'split': train_test})

# TODO
# def fetch_data(chembl_targetid='CHEMBL2047', endpoints=['EC50']):
#     """Download and prep the data from CHEMBL. Throws out duplicates, problematic molecules, and extreme outliers"""
#     from old.MoleculeACE.benchmark.data_processing.preprocessing.data_fetching import main_curator
#     import os
#
#     # fetch + curate data
#     data = main_curator.main(chembl_targetid=chembl_targetid, endpoints=endpoints)
#     # write to Data directory
#     filename = os.path.join('Data', f"{chembl_targetid}_{'_'.join(endpoints)}.csv")
#     data.to_csv(filename)


def find_stereochemical_siblings(smiles: List[str]):
    """ Detects molecules that have different SMILES strings, but ecode for the same molecule with
    different stereochemistry. For racemic mixtures it is often unclear which one is measured/active

    Args:
        smiles: (lst) list of SMILES strings

    Returns: (lst) List of SMILES having a similar molecule with different stereochemistry

    """
    from MoleculeACE.benchmark.cliffs import get_tanimoto_matrix

    lower = np.tril(get_tanimoto_matrix(smiles, radius=4, nBits=4096), k=0)
    identical = np.where(lower == 1)
    identical_pairs = [[smiles[identical[0][i]], smiles[identical[1][i]]] for i, j in enumerate(identical[0])]

    return list(set(sum(identical_pairs, [])))


def augment(smiles, *args, augment_factor=10, max_smiles_len=200):
    """ Augment SMILES strings by adding non-canonical SMILES. Keeps corresponding activity values/CHEMBL IDs """
    augmented_smiles = []
    augmented_args = [[] for _ in args]
    for i, smi in enumerate(tqdm(smiles)):
        generated = smile_augmentation(smi, augment_factor - 1, max_smiles_len)
        augmented_smiles.append(smi)
        augmented_smiles.extend(generated)

        for a, arg in enumerate(args):
            for _ in range(len(generated)+1):
                augmented_args[a].append(arg[i])

    return tuple([augmented_smiles],) + tuple(augmented_args)


def random_smiles(mol):
    """ Generate a random non-canonical SMILES string from a molecule"""
    # https://github.com/michael1788/virtual_libraries/blob/master/experiments/do_data_processing.py
    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, mol.GetNumAtoms()))
    random.shuffle(idxs)
    for i, v in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(mol)


def smile_augmentation(smile, augmentation, max_len):
    """Generate n random non-canonical SMILES strings from a SMILES string with length constraints"""
    # https://github.com/michael1788/virtual_libraries/blob/master/experiments/do_data_processing.py
    mol = Chem.MolFromSmiles(smile)
    s = set()
    for i in range(1000):
        smiles = random_smiles(mol)
        if len(smiles) <= max_len:
            s.add(smiles)
            if len(s) == augmentation:
                break

    return list(s)

