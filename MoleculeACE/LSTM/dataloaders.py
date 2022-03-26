"""
Dataloader for the LSTM models
Derek van Tilborg, Eindhoven University of Technology, March 2022

Inspired by:

Moret, M., Grisoni, F., Katzberger, P. & Schneider, G.
Perplexity-based molecule ranking and bias estimation of chemical language models.
ChemRxiv (2021) doi:10.26434/chemrxiv-2021-zv6f1-v2.
"""

import os
from tensorflow.keras.utils import Sequence
import numpy as np
from MoleculeACE.benchmark.utils import get_config
from MoleculeACE.benchmark.utils.const import CONFIG_PATH_SMILES, RANDOM_SEED

smiles_encoding = get_config(CONFIG_PATH_SMILES)


class DataGeneratorNextToken(Sequence):
    """Generates one-hot encoded smiles + next token data for Keras"""

    def __init__(self, encoded_smiles, batch_size, max_len_model, n_chars,
                 indices_token, token_indices, shuffle=True):
        """Initialization"""
        self.max_len_model = max_len_model
        self.batch_size = batch_size
        self.encoded_smiles = encoded_smiles
        self.shuffle = shuffle
        self.n_chars = n_chars

        self.on_epoch_end()

        self.indices_token = indices_token
        self.token_indices = token_indices

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.encoded_smiles) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.encoded_smiles))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates batch of data containing batch_size samples"""

        switch = 1
        y = np.empty((self.batch_size, self.max_len_model - switch, self.n_chars), dtype=int)
        x = np.empty((self.batch_size, self.max_len_model - switch, self.n_chars), dtype=int)

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            smi = self.encoded_smiles[ID]
            x[i] = smi[:-1]
            y[i] = smi[1:]

        return x, y


class DataGeneratorRegression(Sequence):
    """Generates one-hot encoded smiles + regression data for Keras"""

    def __init__(self, encoded_smiles, activities, batch_size, max_len_model, n_chars,
                 indices_token, token_indices, shuffle=True):

        """Initialization"""
        self.max_len_model = max_len_model
        self.batch_size = batch_size
        self.encoded_smiles = encoded_smiles
        self.activities = activities
        self.shuffle = shuffle
        self.n_chars = n_chars

        self.on_epoch_end()

        self.indices_token = indices_token
        self.token_indices = token_indices

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.encoded_smiles) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        x, y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.encoded_smiles))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates batch of data containing batch_size samples"""
        switch = 1
        y = np.empty((self.batch_size, 1), dtype=float)
        x = np.empty((self.batch_size, self.max_len_model - switch, self.n_chars), dtype=int)

        # Generate data
        for i, ID in enumerate(indexes):
            x[i] = self.encoded_smiles[ID][:-1]
            y[i] = self.activities[ID]

        return x, y


def prep_smiles_pretrain(val_split=0.1, augmentation=10, extra_smiles_train=None, extra_smiles_test=None):
    """

    Args:
        val_split: (int) splitting ratio for the validation data (default=0.1)
        augmentation: (int) augment SMILES n times (default=10)
        random_state: (int) random seed (default=42)
        k: (int) Use k clusters to distribute compounds over train/val (default=10)
        extra_smiles_train: (lst) List of SMILES strings you can add to the train
        extra_smiles_test:  (lst) List of SMILES strings you can add to the test

    Returns: onehot encoded train, test, val

    """
    import pandas as pd
    from MoleculeACE.benchmark import data_processing
    from MoleculeACE.benchmark.data_processing.preprocessing.data_prep import split_smiles

    train_csvs = [dat for dat in os.listdir(os.path.join('Data', 'train')) if dat.startswith('CHEMBL')]
    test_csvs = [dat for dat in os.listdir(os.path.join('Data', 'test')) if dat.startswith('CHEMBL')]

    # Fetch all SMILES strings from train and test files
    smiles_train, smiles_test = [], []
    for file in train_csvs:
         smiles_train.extend(pd.read_csv(os.path.join('Data', 'train', file))['smiles'])
    for file in test_csvs:
         smiles_test.extend(pd.read_csv(os.path.join('Data', 'test', file))['smiles'])

    # Add extra smiles from the user if specified
    if extra_smiles_train is not None:
        smiles_train.extend(extra_smiles_train)
    if extra_smiles_test is not None:
        smiles_train.extend(extra_smiles_test)

    # Remove duplicates and check if test compounds are not in train
    smiles_train = list(set(smiles_train))
    smiles_test = list(set(smiles_test))
    smiles_test = [smi for smi in smiles_test if smi not in smiles_train]

    # Split train into train+validation smiles based on Murcko scaffolds
    train_idx, val_idx = split_smiles(smiles_train, test_split=val_split, clustering_cutoff=0.4)
    smi_tr = [smiles_train[i] for i in train_idx]
    smi_val = [smiles_train[i] for i in val_idx]

    # Perform augmentation of the train and validation data
    max_len = smiles_encoding['max_smiles_len']
    smi_tr_aug, _, _ = data_processing.augment(smi_tr, augment_factor=augmentation, max_smiles_len=max_len,
                                               rand_seed=RANDOM_SEED)
    smi_val_aug, _, _ = data_processing.augment(smi_val, augment_factor=augmentation, max_smiles_len=max_len,
                                                rand_seed=RANDOM_SEED)
    # Don't augment the test SMILES
    smi_test_aug = smiles_test

    # One hot encode everything
    smi_tr_aug_one_hot = data_processing.smiles_to_onehot(smi_tr_aug)
    smi_val_aug_one_hot = data_processing.smiles_to_onehot(smi_val_aug)
    smi_test_aug_one_hot = data_processing.smiles_to_onehot(smi_test_aug)

    # Give a summary of the shape of your data
    print(f"Train: {smi_tr_aug_one_hot[0].shape}, failed {len(smi_tr_aug_one_hot[1])}\n"
          f"Val: {smi_val_aug_one_hot[0].shape}, failed {len(smi_val_aug_one_hot[1])}\n"
          f"Test: {smi_test_aug_one_hot[0].shape}, failed {len(smi_test_aug_one_hot[1])}")

    return smi_tr_aug_one_hot[0], smi_test_aug_one_hot[0], smi_val_aug_one_hot[0]
