"""
Author: Derek van Tilborg -- TU/e -- 22-05-2022

A collection of data-prepping functions
    - split_data():             split ChEMBL csv into train/test taking similarity and cliffs into account. If you want
                                to process your own data, use this function
    - process_data():           see split_data()
    - load_data():              load a pre-processed dataset from the benchmark
    - fetch_data():             download molecular bioactivity data from ChEMBL for a specific drug target

"""

from MoleculeACE.benchmark.cliffs import ActivityCliffs, get_tanimoto_matrix
from MoleculeACE.benchmark.const import RANDOM_SEED
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from typing import List
import pandas as pd
import numpy as np


def split_data(smiles: List[str], bioactivity: List[float], n_clusters: int = 5, test_size: float = 0.2,
               similarity: float = 0.9, potency_fold: int = 10, remove_stereo: bool = True):
    """ Split data into train/test according to activity cliffs and compounds characteristics.

    :param smiles: (List[str]) list of SMILES strings
    :param bioactivity: (List[float]) list of bioactivity values
    :param n_clusters: (int) number of clusters the data is split into for getting homogeneous data splits
    :param test_size: (float) test split
    :param similarity:  (float) similarity threshold for calculating activity cliffs
    :param potency_fold: (float) potency difference threshold for calculating activity cliffs
    :param remove_stereo: (bool) Remove racemic mixtures altogether?

    :return: df[smiles, exp_mean [nM], y, cliff_mol, split]
    """

    original_smiles = smiles
    original_bioactivity = bioactivity

    if remove_stereo:
        stereo_smiles_idx = [smiles.index(i) for i in find_stereochemical_siblings(smiles)]
        smiles = [smi for i, smi in enumerate(smiles) if i not in stereo_smiles_idx]
        bioactivity = [act for i, act in enumerate(bioactivity) if i not in stereo_smiles_idx]
        if len(stereo_smiles_idx) > 0:
            print(f"Removed {len(stereo_smiles_idx)} stereoisomers")

    check_matching(original_smiles, original_bioactivity, smiles, bioactivity)

    y_log = -np.log10(bioactivity)

    cliffs = ActivityCliffs(smiles, bioactivity)
    cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, similarity=similarity, potency_fold=potency_fold)

    check_cliffs(cliffs)

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

    # Check if there is any intersection between train and test molecules
    assert len(np.intersect1d(train_idx, test_idx)) == 0, 'train and test intersect'
    assert len(np.intersect1d(np.array(smiles)[np.where(np.array(train_test) == 'train')],
                              np.array(smiles)[np.where(np.array(train_test) == 'test')])) == 0, \
        'train and test intersect'

    df_out = pd.DataFrame({'smiles': smiles,
                         'exp_mean [nM]': bioactivity,
                         'y': y_log,
                         'cliff_mol': cliff_mols,
                         'split': train_test})

    return df_out


def process_data(smiles: List[str], bioactivity: List[float], n_clusters: int = 5, test_size: float = 0.2,
                 similarity: float = 0.9, potency_fold: int = 10, remove_stereo: bool = False):
    """ Split data into train/test according to activity cliffs and compounds characteristics.

    :param smiles: (List[str]) list of SMILES strings
    :param bioactivity: (List[float]) list of bioactivity values
    :param n_clusters: (int) number of clusters the data is split into for getting homogeneous data splits
    :param test_size: (float) test split
    :param similarity:  (float) similarity threshold for calculating activity cliffs
    :param potency_fold: (float) potency difference threshold for calculating activity cliffs
    :param remove_stereo: (bool) Remove racemic mixtures altogether?

    :return: df[smiles, exp_mean [nM], y, cliff_mol, split]
    """
    return split_data(smiles, bioactivity, n_clusters, test_size, similarity,  potency_fold, remove_stereo)


def fetch_data(chembl_targetid='CHEMBL2047', endpoints=['EC50']):
    """Download and prep the data from CHEMBL. Throws out duplicates, problematic molecules, and extreme outliers"""
    from benchmark.data_fetching import main_curator
    import os

    # fetch + curate data
    data = main_curator.main(chembl_targetid=chembl_targetid, endpoints=endpoints)
    # write to Data directory
    filename = os.path.join('Data', f"{chembl_targetid}_{'_'.join(endpoints)}.csv")
    data.to_csv(filename)


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


def check_matching(original_smiles, original_bioactivity, smiles, bioactivity):
    assert len(smiles) == len(bioactivity), "length doesn't match"
    for smi, label in zip(original_smiles, original_bioactivity):
        if smi in smiles:
            assert bioactivity[smiles.index(smi)] == label, f"{smi} doesn't match label {label}"


def check_cliffs(cliffs, n: int = 10):
    from MoleculeACE.benchmark.cliffs import is_cliff

    # Find the location of 10 random cliffs and check if they are actually cliffs
    m = n
    if np.sum(cliffs.cliffs) < 2*n:
        n = int(np.sum(cliffs.cliffs)/2)

    cliff_loc = np.where(cliffs.cliffs == 1)
    random_cliffs = np.random.randint(0, len(cliff_loc[0]), n)
    cliff_loc = [(cliff_loc[0][c], cliff_loc[1][c]) for c in random_cliffs]

    for i, j in cliff_loc:
        assert is_cliff(cliffs.smiles[i], cliffs.smiles[j], cliffs.bioactivity[i], cliffs.bioactivity[j])

    if len(cliffs.cliffs)-n < m:
        m = len(cliffs.cliffs)-n
    # Find the location of 10 random non-cliffs and check if they are actually non-cliffs
    non_cliff_loc = np.where(cliffs.cliffs == 0)
    random_non_cliffs = np.random.randint(0, len(non_cliff_loc[0]), m)
    non_cliff_loc = [(non_cliff_loc[0][c], non_cliff_loc[1][c]) for c in random_non_cliffs]

    for i, j in non_cliff_loc:
        assert not is_cliff(cliffs.smiles[i], cliffs.smiles[j], cliffs.bioactivity[i], cliffs.bioactivity[j])

