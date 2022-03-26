"""
Code for loading pre-processed data
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

import numpy as np
import pandas as pd
import os

from MoleculeACE.benchmark.utils.const import Descriptors, DATA_PATH, WORKING_DIR, setup_working_dir, datasets
from MoleculeACE.benchmark import data_processing


def load_data(dataset: str = None, train_data: str = None, test_data: str = None, working_dir: str = WORKING_DIR,
              descriptor: Descriptors = Descriptors.ECFP, smiles_colname: str = 'smiles',
              y_colname: str = 'exp_mean [nM]', chembl_id_colname: str = 'chembl_id',
              data_root=DATA_PATH, tolog10: bool = True, fold_threshold: int = 10, similarity_threshold: float = 0.9,
              scale: bool = True, augment_smiles: int = 0, calc_cliffs: bool = True):
    """ Load pre-split and pre-processed data into a Data object

    Args:

        augment_smiles: (int)
        dataset: (str) Dataset Identifier, i.e. 'CHEMBL2047_EC50'
        train_data: (str) path to a train csv. Should contain a column of SMILES and labels
        test_data: (str) path to a test csv. Should contain a column of SMILES and labels
        descriptor: (Descriptors) Descriptor to use for encoding molecules. Pick from: "ECFP/morgan2/circular", "rdkit",
                    "atom_pair", "topological_torsion", "smarts", "maccs", "physchem/drug_like/physicochemical",
                    "3D_WHIM/WHIM", "one_hot/LSTM", "graph"
        smiles_colname: (str) Column name containing SMILES strings in the data, default = 'smiles'
        y_colname: (str) Column name containing bioactivity in the data, default = 'exp_mean [nM]'
        chembl_id_colname: (str) Column name containing bioactivity in the data, default = 'chembl_id'
        tolog10: (bool) Do you want to transform your bioactivity to log10, default = True
        fold_threshold: (int) how much must the fold change be before being an activity cliff? (default=10)
        similarity_threshold: (float) how similar must two compounds be before being an activity cliff (default=0.7)
        scale: (bool) Scale the data? default = True
        calc_cliffs: (bool) Calculate activity cliffs??

    Returns: MoleculeACE.Utils.data_prep.Data object

    """
    setup_working_dir(working_dir)

    # 1. Read data, either an included one or custom train/test data
    if dataset in datasets:
        train_dat = pd.read_csv(os.path.join(WORKING_DIR, 'benchmark_data', 'train', f"{dataset}_train.csv"))
        test_dat = pd.read_csv(os.path.join(WORKING_DIR, 'benchmark_data', 'test', f"{dataset}_test.csv"))
    elif train_data is None and test_data is None:
        train_dat = pd.read_csv(os.path.join(working_dir, 'benchmark_data', 'train', f"{dataset}_train.csv"))
        test_dat = pd.read_csv(os.path.join(working_dir, 'benchmark_data', 'test', f"{dataset}_test.csv"))
    elif train_data is not None and test_data is not None:
        train_dat = pd.read_csv(train_data)
        test_dat = pd.read_csv(test_data)
        if dataset is None:
            dataset = 'custom_data'
    else:
        print('Provide either an included data set or custom ones in train_data, test_data (as a .csv file)')
        print(f"Included data sets to pick from: {datasets}")
        raise ValueError('Could not find any data')

    if chembl_id_colname is None:
        chembl_id_colname = smiles_colname

    # 2. Put everything in a Data class
    data = data_processing.Data(name=dataset,
                                smiles_train=train_dat[smiles_colname].to_list(),
                                y_train=train_dat[y_colname].to_list(),
                                chembl_id_train=train_dat[chembl_id_colname].to_list(),
                                smiles_test=test_dat[smiles_colname].to_list(),
                                y_test=test_dat[y_colname].to_list(),
                                chembl_id_test=test_dat[chembl_id_colname].to_list(),
                                working_dir=working_dir)

    # 3. Convert to log if needed
    if tolog10:
        data.y_train = [-i for i in np.log10(data.y_train).tolist()]
        data.y_test = [-i for i in np.log10(data.y_test).tolist()]
        data.log10 = tolog10

    # 4. Search for activity cliffs
    if calc_cliffs:
        print("\nSearching for activity cliffs..")
        data.get_cliffs(fold_threshold=fold_threshold, similarity_threshold=similarity_threshold)

    # 5. Encode data
    # augment if needed. Only useful for LSTM data. Other descriptors will just yield duplicates
    if augment_smiles > 0:
        print(f"Augmenting SMILES {augment_smiles}x ...")
        data.augment_smiles(augmentation=augment_smiles)

    # Encode data with the chosen descriptor
    print(f"Encoding molecules with {descriptor.value} ...")
    data.redo_encoding(descriptor)

    # 6. Scale data
    if scale:
        print(f"Scaling data ...")
        data.scale_data()

    return data
