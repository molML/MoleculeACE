"""
A collection of random functions that are used in different places
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

from MoleculeACE.benchmark.cliffs import ActivityCliffs
from MoleculeACE.benchmark.const import Descriptors, datasets, DATA_PATH, RANDOM_SEED, CONFIG_PATH_SMILES
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.model_selection import StratifiedKFold
from yaml import load, Loader, dump
from typing import List, Union
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import torch
import json
import os
import re
import random
from rdkit import Chem


class Data:
    def __init__(self, filename: str):
        """ Data class to easily load and featurize molecular bioactivity data

        :param filename: (str) Filename of a .csv file with the following columns: smiles -- containing SMILES strings,
        y -- containing labels, cliff_mol -- column with 1 if cliff compounds else 0, split -- column with 'train' if in
        training split else 'test'. Either a full path to the file or the name of one of the included datasets works.
        """

        if filename in datasets:
            filename = os.path.join(DATA_PATH, f"{filename}.csv")
        df = pd.read_csv(filename)

        self.smiles_train = df[df['split'] == 'train']['smiles'].tolist()
        self.y_train = df[df['split'] == 'train']['y'].tolist()
        self.cliff_mols_train = df[df['split'] == 'train']['cliff_mol'].tolist()

        self.smiles_test = df[df['split'] == 'test']['smiles'].tolist()
        self.y_test = df[df['split'] == 'test']['y'].tolist()
        self.cliff_mols_test = df[df['split'] == 'test']['cliff_mol'].tolist()

        from MoleculeACE.benchmark.featurization import Featurizer
        self.featurizer = Featurizer()

        self.x_train = None
        self.x_test = None

        self.featurized_as = 'Nothing'
        self.augmented = 0

    def featurize_data(self, descriptor: Descriptors, **kwargs):
        if descriptor is Descriptors.PHYSCHEM or descriptor is Descriptors.WHIM:
            self.x_train = self.featurizer(descriptor, smiles=self.smiles_train, scale=True, **kwargs)
            self.x_test = self.featurizer(descriptor, smiles=self.smiles_test, scale_test_on_train=True, **kwargs)
        else:
            self.x_train = self.featurizer(descriptor, smiles=self.smiles_train, **kwargs)
            self.x_test = self.featurizer(descriptor, smiles=self.smiles_test, **kwargs)
        self.featurized_as = descriptor.name

    def shuffle(self):
        c = list(zip(self.smiles_train, self.y_train, self.cliff_mols_train))  # Shuffle all lists together
        random.shuffle(c)
        self.smiles_train, self.y_train, self.cliff_mols_train = zip(*c)

        self.smiles_train = list(self.smiles_train)
        self.y_train = list(self.y_train)
        self.cliff_mols_train = list(self.cliff_mols_train)

    def augment(self, augment_factor: int = 10, max_smiles_len: int = 200):
        self.smiles_train, self.y_train, self.cliff_mols_train = augment(self.smiles_train, self.y_train,
                                                                         self.cliff_mols_train,
                                                                         augment_factor=augment_factor,
                                                                         max_smiles_len=max_smiles_len)
        self.augmented = augment_factor

    def __call__(self, descriptor: Descriptors, **kwargs):
        self.featurize_data(descriptor, **kwargs)

    def __repr__(self):
        return f"Data object with molecules as: {self.featurized_as}. {len(self.y_train)} train/{len(self.y_test)} test"


def get_config(file: str):
    """ Load a yml config file"""
    if file.endswith('.yml') or file.endswith('.yaml'):
        with open(file, "r", encoding="utf-8") as read_file:
            config = load(read_file, Loader=Loader)
    if file.endswith('.json'):
        with open(file, 'r') as f:
            config = json.load(f)
    return config


def write_config(filename: str, args: dict):
    """ Write a dictionary to a .yml file"""
    args = {k: v.item() if isinstance(v, np.generic) else v for k, v in args.items()}
    with open(filename, 'w') as file:
        documents = dump(args, file)


def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    import numpy as np

    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))


def calc_cliff_rmse(y_test_pred: Union[List[float], np.array], y_test: Union[List[float], np.array],
                    cliff_mols_test: List[int] = None, smiles_test: List[str] = None,
                    y_train: Union[List[float], np.array] = None, smiles_train: List[str] = None, **kwargs):
    """ Calculate the RMSE of activity cliff compounds """

    # Check if we can compute activity cliffs when pre-computed ones are not provided.
    if cliff_mols_test is None:
        if smiles_test is None or y_train is None or smiles_train is None:
            raise ValueError('if cliff_mols_test is None, smiles_test, y_train, and smiles_train should be provided '
                             'to compute activity cliffs')

    # Convert to numpy array if it is none
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    if cliff_mols_test is None:
        y_train = np.array(y_train) if type(y_train) is not np.array else y_train
        # Calculate cliffs and
        cliffs = ActivityCliffs(smiles_train + smiles_test, np.append(y_train, y_test))
        cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, **kwargs)
        # Take only the test cliffs
        cliff_mols_test = cliff_mols[len(smiles_train):]

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)


def cross_validate(model, data, n_folds: int = 5, early_stopping: int = 10, seed: int = RANDOM_SEED,
                   save_path: str = None, **hyperparameters):

    x_train = data.x_train
    y_train = data.y_train
    x_test = data.x_test
    y_test = data.y_test

    ss = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    cutoff = np.median(y_train)
    labels = [0 if i < cutoff else 1 for i in y_train]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(labels, labels)]

    rmse_scores = []
    cliff_rmse_scores = []
    for i_split, split in enumerate(tqdm(splits)):

        # Convert numpy types to regular python type (this bug cost me ages)
        hyperparameters = {k: v.item() if isinstance(v, np.generic) else v for k, v in hyperparameters.items()}

        f = model(**hyperparameters)

        if type(x_train) is BatchEncoding:
            x_tr_fold = {'input_ids': x_train['input_ids'][split['train_idx']],
                         'attention_mask': x_train['attention_mask'][split['train_idx']]}
            x_val_fold = {'input_ids': x_train['input_ids'][split['val_idx']],
                         'attention_mask': x_train['attention_mask'][split['val_idx']]}
        else:
            x_tr_fold = [x_train[i] for i in split['train_idx']] if type(x_train) is list else x_train[split['train_idx']]
            x_val_fold = [x_train[i] for i in split['val_idx']] if type(x_train) is list else x_train[split['val_idx']]

        y_tr_fold = [y_train[i] for i in split['train_idx']] if type(y_train) is list else y_train[split['train_idx']]
        y_val_fold = [y_train[i] for i in split['val_idx']] if type(y_train) is list else y_train[split['val_idx']]

        f.train(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, early_stopping)

        # Save model to "save_path+_{fold}.pkl"
        if save_path is not None:
            with open(f"{save_path.split('.')[-2]}_{i_split}.{save_path.split('.')[-1]}", 'wb') as handle:
                pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)

        y_hat = f.predict(x_test)

        rmse = calc_rmse(y_test, y_hat)
        rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=y_test,
                                     cliff_mols_test=data.cliff_mols_test)

        rmse_scores.append(rmse)
        cliff_rmse_scores.append(rmse_cliff)

        del f
        torch.cuda.empty_cache()

    # Return the rmse and cliff rmse for all folds
    return rmse_scores, cliff_rmse_scores


def smi_tokenizer(smi: str):
    """
    Tokenize a SMILES
    """
    pattern = "(\[|\]|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Si|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    return tokens


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


smiles_encoding = get_config(CONFIG_PATH_SMILES)


def smile_augmentation(smile, augmentation, max_len):
    """Generate n random non-canonical SMILES strings from a SMILES string with length constraints"""
    # https://github.com/michael1788/virtual_libraries/blob/master/experiments/do_data_processing.py
    mol = Chem.MolFromSmiles(smile)
    s = set()
    for i in range(1000):
        if len(s) == augmentation:
            break

        smiles = random_smiles(mol)
        if len(smiles) <= max_len:
            tokens = smi_tokenizer(smiles)
            if all([tok in smiles_encoding['token_indices'] for tok in tokens]):
                s.add(smiles)

    return list(s)