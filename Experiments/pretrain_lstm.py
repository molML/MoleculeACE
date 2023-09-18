"""
Author: Derek van Tilborg -- TU/e -- 08-06-2022

Script to pretrain a LSTM using next-token-prediction. We used 10x data augmentation, 100 epochs with early stopping,
and 10% validation data. The weights of this model are later used to initialize all LSTM regression models.

"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from MoleculeACE.benchmark.const import datasets, DATA_PATH, WORKING_DIR, CONFIG_PATH_SMILES, RANDOM_SEED
from MoleculeACE.benchmark.cliffs import get_tanimoto_matrix
from MoleculeACE.benchmark.utils import augment, get_config
from MoleculeACE.models.lstm import LSTMNextToken
from MoleculeACE.benchmark.featurization import Featurizer
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from typing import List
import pandas as pd
import numpy as np
import pickle
import os


smiles_encoding = get_config(CONFIG_PATH_SMILES)
AUGMENT = 10  # Augment SMILES strings n times
N_CLUSTERS = 10  # Cluster molecules into n clusters and use those clusters to get homogeneous train/val splits
VAL_SPLIT = 0.1  # Validation split
EPOCHS = 100  # Train for n epochs
SAVE_PATH = os.path.join(WORKING_DIR, "pretrained_models", "pretrained_lstm.h5")


def get_all_smiles():

    train_smiles_all = []
    test_smiles = []

    for filename in datasets:
        df = pd.read_csv(os.path.join(DATA_PATH, f"{filename}.csv"))
        train_smiles_all.extend(df[df['split'] == 'train']['smiles'].tolist())
        test_smiles.extend(df[df['split'] == 'test']['smiles'].tolist())

    return train_smiles_all, test_smiles


def split_train_val(smiles: List[str], val_split: float = 0.1,  n_clusters: int = 10, random_state: int = RANDOM_SEED):

    # Perform spectral clustering on a Tanimoto distance matrix
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=random_state, affinity='precomputed')
    clusters = spectral.fit(get_tanimoto_matrix(smiles)).labels_

    train_smiles, val_smiles = train_test_split(smiles, test_size=val_split, random_state=random_state,
                                                stratify=clusters, shuffle=True)

    return train_smiles, val_smiles


def main():

    # Fetch all molecules
    train_smiles, test_smiles = get_all_smiles()

    # remove duplicates
    train_smiles = list(set(train_smiles))
    test_smiles = list(set(test_smiles))

    # split train into train/val
    train_smiles, val_smiles = split_train_val(train_smiles, val_split=VAL_SPLIT, n_clusters=N_CLUSTERS,
                                               random_state=RANDOM_SEED)

    # augment all training/val smiles
    train_smiles = augment(train_smiles, augment_factor=AUGMENT)[0]
    val_smiles = augment(val_smiles, augment_factor=AUGMENT)[0]

    # Train autoregressive LSTM
    model = LSTMNextToken()
    model.train(train_smiles, val_smiles, epochs=EPOCHS, early_stopping_patience=10)

    # Save model
    model.model.save(SAVE_PATH)

    # Test model
    y_hat, y_true = model.test(test_smiles)

    # Don't calculate the accuracy of predicting the padding character. This highly inflates the accuracy.
    pad_char_idx = smiles_encoding['token_indices'][smiles_encoding['pad_char']]
    acc = np.mean([int(t == p) for t, p in zip(y_true, y_hat) if p != pad_char_idx])
    print(f"Accuracy: {acc:.4f}")

    pd.DataFrame({"accuracy": [acc]}).to_csv('pretraining_acc.csv', index=False)


if __name__ == '__main__':
    main()
