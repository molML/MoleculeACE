"""
Class for data processing: prep a csv, calculate activity cliffs, and split data
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

import numpy as np
import pandas as pd
import pickle
import os

from MoleculeACE.benchmark import Cliffs
from MoleculeACE.benchmark import data_processing
from MoleculeACE.benchmark.utils.const import WORKING_DIR, setup_working_dir


def process_data(filename: str, smiles_colname: str = 'smiles', y_colname: str = 'exp_mean [nM]',
                 working_dir: str = WORKING_DIR, chembl_id_colname: str = 'chembl_id', remove_stereo: bool = True,
                 fold_threshold: int = 10, similarity_threshold: float = 0.9, test_size: float = 0.20,
                 clustering_cutoff: float = 0.4):
    """ Process the csv with SMILES and activities: prep data, log10, splitting, cliff calculation (+saving)

    Args:
        filename: (str) Path to filename
        smiles_colname: (str) colname of the SMILES column of your file
        y_colname: (str) colname of the bioactivity column of your file
        chembl_id_colname: (str) colname of the ID column of your file (default=None)
        remove_stereo: (bool) Remove enantiomers? (default=True)
        fold_threshold: (int) potency difference threshold for activity cliffs (default=10)
        similarity_threshold: (float) Structure similarity threshold for activity cliffs (default=0.9)
        test_size: (float) ratio of the test split (default=0.20)
        clustering_cutoff: similarity cutoff for clustering compounds for the train/test split (default=0.4)

    Returns: Data

    """

    setup_working_dir(working_dir)

    # 1. Prep data: load data from file, remove stereochemical redundancies, and log transform
    smiles, activities, chembl_id = data_processing.prep_data(filename, smiles_colname, y_colname, chembl_id_colname,
                                                              remove_stereo)

    # Log10 transform the bioactivity if needed
    activities = np.log10(activities).tolist()

    # split the path of the filename into a name. i.e., ~/documents/my_data.csv --> my_data
    name = filename.split('/')[-1].split('.')[0]

    # Put everything in a Data class
    data = data_processing.Data(name=name, log10=True, working_dir=working_dir)
    data.all_smiles = smiles
    data.all_bioactivities = activities
    if chembl_id_colname is not None:
        data.all_chembl_ids = chembl_id

    # 2. Finding activity cliffs.
    print("\nSearching for activity cliffs..")
    data.cliffs = Cliffs(data.all_smiles, data.all_bioactivities, in_log10=True, fold_threshold=fold_threshold,
                         similarity_threshold=similarity_threshold)

    # 3. Split data
    train_idx, test_idx = data_processing.split_data(data.all_smiles, data.cliffs.cliff_mols_soft_consensus,
                                                     test_size=test_size, clustering_cutoff=clustering_cutoff)

    # Convert back to nM from log10 (for consistency)
    data.smiles_train = [data.all_smiles[i] for i in train_idx]
    data.smiles_test = [data.all_smiles[i] for i in test_idx]
    data.y_train = [10 ** data.all_bioactivities[i] for i in train_idx]
    data.y_test = [10 ** data.all_bioactivities[i] for i in test_idx]
    if chembl_id_colname is not None:
        data.chembl_id_train = [data.all_chembl_ids[i] for i in train_idx]
        data.chembl_id_test = [data.all_chembl_ids[i] for i in test_idx]

    # Subset all cliffs into train and test sets as well
    data.cliffs.smiles = data.smiles_train
    data.cliffs.test_smiles = data.smiles_test
    data.cliffs.train_test_cliffs(train_idx, test_idx)
    print(data.cliffs)

    # Save activity cliffs for later
    cliff_pickle_name = os.path.join(working_dir, 'activity_cliffs',
                                     f"{name}_sim{similarity_threshold}_fold{fold_threshold}.pkl")
    with open(cliff_pickle_name, 'wb') as handle:
        pickle.dump(data.cliffs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved activity cliffs as pickle: {cliff_pickle_name}")

    # Save a test and train csv
    if chembl_id_colname is not None:
        train_df = pd.DataFrame(zip(data.smiles_train, data.chembl_id_train, data.y_train),
                                columns=[smiles_colname, chembl_id_colname, y_colname])
        test_df = pd.DataFrame(zip(data.smiles_test, data.chembl_id_test, data.y_test),
                               columns=[smiles_colname, chembl_id_colname, y_colname])
    else:
        train_df = pd.DataFrame(zip(data.smiles_train, data.y_train),
                                columns=[smiles_colname, y_colname])
        test_df = pd.DataFrame(zip(data.smiles_test, data.y_test),
                               columns=[smiles_colname, y_colname])

    train_df.to_csv(os.path.join(working_dir, 'benchmark_data', 'train', f'{name}_train.csv'))
    test_df.to_csv(os.path.join(working_dir, 'benchmark_data', 'test', f'{name}_test.csv'))
