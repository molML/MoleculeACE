"""
Class that holds all data throughout the MoleculeACE pipeline
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

import os
import pickle
import numpy as np

from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler

from MoleculeACE.benchmark import data_processing
from MoleculeACE.benchmark import Cliffs
from MoleculeACE.benchmark.utils import get_config
from MoleculeACE.benchmark.utils.const import RANDOM_SEED, Descriptors, CONFIG_PATH, WORKING_DIR

smiles_encoding = get_config(os.path.join(CONFIG_PATH, 'default', 'SMILES.yml'))


class Data:
    """ Data class that holds all data"""

    def __init__(self, name: str, log10: bool = True, all_smiles: list = None, all_chembl_ids: list = None,
                 all_bioactivities: list = None, smiles_train: list = None, chembl_id_train: list = None,
                 x_train: np.array = None, y_train: list = None, smiles_test: list = None,
                 chembl_id_test: list = None, x_test: np.array = None, y_test: list = None, smiles_val: list = None,
                 chembl_id_val: list = None, x_val: np.array = None, y_val: list = None, cliffs: Cliffs = None,
                 descriptor: Descriptors = None, working_dir: str = WORKING_DIR):

        self.name = name
        self.log10 = log10
        self.descriptor = descriptor
        self.augmentation = 0
        self.working_dir = working_dir

        self.all_smiles = all_smiles
        self.all_chembl_ids = all_chembl_ids
        self.all_bioactivities = all_bioactivities

        self.smiles_train = smiles_train
        self.chembl_id_train = chembl_id_train
        self.x_train = x_train
        self.y_train = y_train

        self.smiles_test = smiles_test
        self.chembl_id_test = chembl_id_test
        self.x_test = x_test
        self.y_test = y_test

        self.smiles_val = smiles_val
        self.chembl_id_val = chembl_id_val
        self.x_val = x_val
        self.y_val = y_val

        self.cliffs = cliffs
        self.scaler = None

    def augment_smiles(self, augmentation=10, max_smiles_len=smiles_encoding['max_smiles_len'], shuffle=False):
        """ Perform augmentation on the train data (and validation data if applicable) """
        self.augmentation = augmentation
        if augmentation > 0:
            if self.smiles_train is not None:
                self.smiles_train, self.y_train, self.chembl_id_train = data_processing.augment(
                    smiles=self.smiles_train,
                    activity=self.y_train,
                    chembl=self.chembl_id_train,
                    augment_factor=augmentation,
                    max_smiles_len=max_smiles_len,
                    shuffle_data=shuffle)

            if self.smiles_val is not None:
                self.smiles_val, self.y_val, self.chembl_id_val = data_processing.augment(smiles=self.smiles_val,
                                                                                          activity=self.y_val,
                                                                                          chembl=self.chembl_id_val,
                                                                                          augment_factor=augmentation,
                                                                                          max_smiles_len=max_smiles_len,
                                                                                          shuffle_data=False)

    def redo_encoding(self, descriptor: Descriptors):
        """ Compute or re-compute molecule encoding from SMILES"""
        if self.smiles_train is None or self.smiles_test is None:
            raise ValueError('Data.smiles_train and/or Data.smiles_test is missing')

        self.x_train, failed_idx_train = data_processing.encode_smiles(self.smiles_train, descriptor)
        self.x_test, failed_idx_test = data_processing.encode_smiles(self.smiles_test, descriptor)
        if self.smiles_val is not None:
            self.x_val, failed_idx_val = data_processing.encode_smiles(self.smiles_val, descriptor)

        # Remove failed molecules (can happen in rare occasions with WHIM or One-hot encoding)
        for i in sorted(failed_idx_train, reverse=True):
            self.smiles_train.pop(i)
            self.chembl_id_train.pop(i)
            self.y_train.pop(i)

        for i in sorted(failed_idx_test, reverse=True):
            self.smiles_test.pop(i)
            self.chembl_id_train.pop(i)
            self.y_test.pop(i)

        if self.smiles_val is not None:
            for i in sorted(failed_idx_val, reverse=True):
                self.smiles_val.pop(i)
                self.chembl_id_train.pop(i)
                self.y_val.pop(i)

        self.descriptor = descriptor

    def get_cliffs(self, fold_threshold: int = 10, similarity_threshold: float = 0.9):
        """ Find activity cliffs

        Args:
            fold_threshold: (int) how much must the fold change be before being an activity cliff? (default=10)
            similarity_threshold: (float) how similar must two compounds be before being an activity cliff (default=0.7)

        """
        # Load cliffs if they have been calculated before
        cliff_pickle_name = os.path.join(self.working_dir, 'activity_cliffs',
                                         f"{self.name}_sim{similarity_threshold}_fold{fold_threshold}.pkl")
        if os.path.exists(cliff_pickle_name):
            with open(cliff_pickle_name, 'rb') as handle:
                self.cliffs = pickle.load(handle)
        # Otherwise, calculate them
        else:
            os.makedirs(os.path.join(self.working_dir, 'activity_cliffs'), exist_ok=True)
            self.cliffs = Cliffs(smiles=self.smiles_train,
                                 activity=self.y_train,
                                 test_smiles=self.smiles_test,
                                 test_activity=self.y_test,
                                 in_log10=self.log10,
                                 fold_threshold=fold_threshold,
                                 similarity_threshold=similarity_threshold)

            # Save for later
            with open(cliff_pickle_name, 'wb') as handle:
                pickle.dump(self.cliffs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def scale_data(self):
        """
            Scales the train data. Scales the test data according to the same scaling
            New test data must be scaled by Data.scaler.transform(x)
        """
        do_not_scale = [Descriptors.ECFP, Descriptors.MORGAN2, Descriptors.CIRCULAR, Descriptors.RDKIT,
                        Descriptors.ATOM_PAIR, Descriptors.TOPOLOGICAL_TORSION, Descriptors.SMARTS, Descriptors.MACCS,
                        Descriptors.SMILES, Descriptors.LSTM, Descriptors.GRAPH, Descriptors.CANONICAL_GRAPH,
                        Descriptors.ATTENTIVE_GRAPH]

        # only scale some descriptors. Binary data (most fingerprints) is not suitable for scaling
        if self.descriptor in do_not_scale:
            print(f"{self.descriptor} data is not suited for auto scaling")
        else:
            if self.x_train is not None:
                self.scaler = StandardScaler().fit(self.x_train)
                self.x_train = self.scaler.transform(self.x_train)
            else:
                print(f"Provide some train data first (Data.x_train should be a np.array)")
            # Use the scaler of the train data to scale the test (and/or val) data accordingly
            if self.x_test is not None:
                self.x_test = self.scaler.transform(self.x_test)
            if self.x_val is not None:
                self.x_val = self.scaler.transform(self.x_val)

    def get_cv_folds(self, n_splits=5, random_state=RANDOM_SEED, stratified=True):
        """ Gets cross validation folds that can be used as cv in sklearn models. Uses a stratified approach

        Args:
            n_splits: (int) number of folds
            random_state: (int) random seed
            stratified: (bool) use stratified splitting

        Returns: cv, iterable containing fold_i, (train_indices, test_indices)

        """
        if stratified:
            if self.cliffs is None:
                print('Cliffs not provided, looking for them now (with default settings)')
                self.cliffs = Cliffs(self.smiles_train, self.y_train,
                                     self.smiles_test, self.y_test, in_log10=self.log10)

            has_cliffs = [1 if i in self.cliffs.cliff_mols_soft_consensus else 0 for i in self.smiles_train]
            skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            cv_folds = skf.split(self.x_train, has_cliffs)
        else:
            ss = ShuffleSplit(n_splits=n_splits, random_state=random_state)
            cv_folds = ss.split(self.x_train)
        return cv_folds

    def __repr__(self):
        out = f"Data: {self.name}"
        if self.descriptor is not None:
            out = out + f"\nEncoding: {self.descriptor}"
        if self.y_train is not None:
            out = out + f"\nTrain size: {len(self.y_train)}"
        if self.y_test is not None:
            out = out + f"\nTest size: {len(self.y_test)}"
        if self.y_val is not None:
            out = out + f"\nValidation size: {len(self.y_val)}"
        if self.cliffs is not None:
            if self.cliffs.cliff_mols_soft_consensus_tr is not None:
                out = out + f"\nCliff compounds in train: {len(self.cliffs.cliff_mols_soft_consensus_tr)}"
            if self.cliffs.cliff_mols_soft_consensus_tst is not None:
                out = out + f"\nCliff compounds in test: {len(self.cliffs.cliff_mols_soft_consensus_tst)}"
        return out
