"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

A collection of random functions that are used in different places

    - Data:                     Class that holds all data, featurizes it, and can augment it
        - featurize_data()
        - shuffle()
        - augment()
    - get_config():             function to load a dict of parameters from a .yml or .json
    - write_config():           function to write a dict of parameters to a .yml
    - calc_rmse():              calculate Root Mean Square Error
    - calc_cliff_rmse():        calculate Root Mean Square Error on activity cliff compounds
    - cross_validate():         cross-validates a model and calculates RMSE and RMSE_cliff for each fold
    - smi_tokenizer():          convert a SMILES string into character tokens
    - augment():                add non-canonical SMILES strings and manage the labels accordingly
    - random_smiles():          create a single random non-canonical SMILES string
    - smile_augmentation():     create n alternative SMILES strings

"""

from MoleculeACE.benchmark.cliffs import ActivityCliffs
from MoleculeACE.benchmark.const import Descriptors, datasets, DATA_PATH, RANDOM_SEED, CONFIG_PATH_SMILES, CONFIG_PATH
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.model_selection import StratifiedKFold
from yaml import load, Loader, dump
from typing import List, Union
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import numpy as np
import pickle
import torch
import json
import os
import re
import random
import warnings
import tensorflow as tf


class Data:
    def __init__(self, file: Union[str, pd.DataFrame]):
        """ Data class to easily load and featurize molecular bioactivity data

        :param file: 1. (str) path to .csv file with the following columns; 'smiles': (str) SMILES strings
                                                                                'y': (float) bioactivity values
                                                                                'cliff_mol': (int) 1 if cliff else 0
                                                                                'split': (str) 'train' or 'test'
                     2. (str) name of a dataset provided with the benchmark; see MoleculeACE.benchmark.const.datasets
                     3. (pandas.DataFrame) pandas dataframe with columns similar to 1. (see above)
        """

        # Either load a .csv file or use a provided dataframe
        if type(file) is str:
            if file in datasets:
                file = os.path.join(DATA_PATH, f"{file}.csv")
            df = pd.read_csv(file)
        else:
            df = file

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
        """ Encode your molecules with Descriptors.ECFP, Descriptors.MACCS, Descriptors.WHIM, Descriptors.PHYSCHEM,
        Descriptors.GRAPH, Descriptors.SMILES, Descriptors.TOKENS """

        if descriptor is Descriptors.PHYSCHEM or descriptor is Descriptors.WHIM:
            self.x_train = self.featurizer(descriptor, smiles=self.smiles_train, scale=True, **kwargs)
            self.x_test = self.featurizer(descriptor, smiles=self.smiles_test, scale_test_on_train=True, **kwargs)
        else:
            self.x_train = self.featurizer(descriptor, smiles=self.smiles_train, **kwargs)
            self.x_test = self.featurizer(descriptor, smiles=self.smiles_test, **kwargs)
        self.featurized_as = descriptor.name

    def shuffle(self):
        """ Shuffle training data """
        c = list(zip(self.smiles_train, self.y_train, self.cliff_mols_train))  # Shuffle all lists together
        random.shuffle(c)
        self.smiles_train, self.y_train, self.cliff_mols_train = zip(*c)

        self.smiles_train = list(self.smiles_train)
        self.y_train = list(self.y_train)
        self.cliff_mols_train = list(self.cliff_mols_train)

    def augment(self, augment_factor: int = 10, max_smiles_len: int = 200):
        """ Augment training SMILES strings n times (Do this before featurizing them please)"""
        self.smiles_train, self.y_train, self.cliff_mols_train = augment(self.smiles_train, self.y_train,
                                                                         self.cliff_mols_train,
                                                                         augment_factor=augment_factor,
                                                                         max_smiles_len=max_smiles_len)
        if self.x_train is not None:
            if len(self.y_train) > len(self.x_train):
                warnings.warn("DON'T FORGET TO RE-FEATURIZE YOUR AUGMENTED DATA")
        self.augmented = augment_factor

    def __call__(self, descriptor: Descriptors, **kwargs):
        self.featurize_data(descriptor, **kwargs)

    def __repr__(self):
        return f"Data object with molecules as: {self.featurized_as}. {len(self.y_train)} train/{len(self.y_test)} test"


def load_model(filename: str):
    """ Load a model """
    if filename.endswith('.h5'):
        from tensorflow.keras.models import load_model
        model = load_model(filename)
    else:
        with open(filename, 'rb') as handle:
            model = pickle.load(handle)
    return model


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


def get_benchmark_config(dataset: str, algorithm, descriptor):
    """ Get the default configs

    :param dataset: (str) one of the 30 datasets included in MoleculeACE
    :param algorithm: (str) Algorithm class or name of one of the supported algorithms. Pick from:
        ['RF', 'SVM', 'GBM', 'KNN', 'MLP', 'GCN', 'MPNN', 'AFP', 'GAT', 'CNN', 'LSTM', 'Transformer']
    :param descriptor: (str) Descriptor class or name of one of the supported descriptors. Pick from:
        ['ECFP', 'MACCS', 'PHYSCHEM', 'WHIM', 'GRAPH', 'SMILES', 'TOKENS']
    :return: (dict) hyperparameters
    """

    from MoleculeACE.models import RF, SVM, GBM, KNN, MLP, GCN, MPNN, AFP, GAT, CNN, LSTM, Transformer
    list_of_algos = [RF, SVM, GBM, KNN, MLP, GCN, MPNN, AFP, GAT, CNN, LSTM, Transformer]

    # Dataset
    if dataset not in datasets:
        raise ValueError(f"Chosen dataset is not included, please pick from: {datasets}")

    # Descriptor
    if type(descriptor) is not str:
        if not descriptor.name in [i.name for i in Descriptors]:
            raise ValueError(
                f"Chosen descriptor is not supported, please pick from: {[i.__str__() for i in Descriptors]}")
        descriptor = descriptor.name
    else:
        if not descriptor in [i.name for i in Descriptors]:
            raise ValueError(f"Chosen descriptor is not supported, please pick from: {[i.name for i in Descriptors]}")

    # Algorithm
    if type(algorithm) is not str:
        if algorithm not in list_of_algos:
            raise ValueError(f"Chosen algorithm is not supported, please pick from: {list_of_algos}")
        algorithm = algorithm.__name__
    else:
        if algorithm not in [i.__name__ for i in list_of_algos]:
            raise ValueError(
                f"Chosen algorithm is not supported, please pick from: {[i.__name__ for i in list_of_algos]}")

    combinations = {'ECFP': ['RF', 'SVM', 'GBM', 'KNN', 'MLP'],
                    'MACCS': ['RF', 'SVM', 'GBM', 'KNN'],
                    'PHYSCHEM': ['RF', 'SVM', 'GBM', 'KNN'],
                    'WHIM': ['RF', 'SVM', 'GBM', 'KNN'],
                    'GRAPH': ['GCN', 'MPNN', 'AFP', 'GAT'],
                    'TOKENS': ['Transformer'],
                    'SMILES': ['CNN', 'LSTM']}

    if algorithm not in combinations[descriptor]:
        raise ValueError(f'Given combination of descriptor and algorithm is not supported. Pick from: {combinations}')

    config_path = os.path.join(CONFIG_PATH, 'benchmark', dataset, f"{algorithm}_{descriptor}.yml")
    hyperparameters = get_config(config_path)

    return hyperparameters


def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))


def calc_cliff_rmse(y_test_pred: Union[List[float], np.array], y_test: Union[List[float], np.array],
                    cliff_mols_test: List[int] = None, smiles_test: List[str] = None,
                    y_train: Union[List[float], np.array] = None, smiles_train: List[str] = None, **kwargs):
    """ Calculate the RMSE of activity cliff compounds

    :param y_test_pred: (lst/array) predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :param smiles_test: (lst) list of SMILES strings of the test molecules
    :param y_train: (lst/array) train labels
    :param smiles_train: (lst) list of SMILES strings of the train molecules
    :param kwargs: arguments for ActivityCliffs()
    :return: float RMSE on activity cliff compounds
    """

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
    """

    :param model: a model that has a train(), test(), and predict() method and is initialized with its hyperparameters
    :param data: Moleculace.benchmark.utils.Data object
    :param n_folds: (int) n folds for cross-validation
    :param early_stopping: (int) stop training when not making progress for n epochs
    :param seed: (int) random seed
    :param save_path: (str) path to save trained models
    :param hyperparameters: (dict) dict of hyperparameters {name_of_param: value}

    :return: (list) rmse_scores, (list) cliff_rmse_scores
    """

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
            x_tr_fold = [x_train[i] for i in split['train_idx']] if type(x_train) is list else x_train[
                split['train_idx']]
            x_val_fold = [x_train[i] for i in split['val_idx']] if type(x_train) is list else x_train[split['val_idx']]

        y_tr_fold = [y_train[i] for i in split['train_idx']] if type(y_train) is list else y_train[split['train_idx']]
        y_val_fold = [y_train[i] for i in split['val_idx']] if type(y_train) is list else y_train[split['val_idx']]

        f.train(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, early_stopping)

        # Save model to "save_path+_{fold}.pkl"
        if save_path is not None:
            if save_path.endswith('.h5'):
                f.model.save(f"{save_path.split('.')[-2]}_{i_split}.{save_path.split('.')[-1]}")
            else:
                with open(f"{save_path.split('.')[-2]}_{i_split}.{save_path.split('.')[-1]}", 'wb') as handle:
                    pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)

        y_hat = f.predict(x_test)

        rmse = calc_rmse(y_test, y_hat)
        rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=y_test,
                                     cliff_mols_test=data.cliff_mols_test)

        rmse_scores.append(rmse)
        cliff_rmse_scores.append(rmse_cliff)

        del f.model
        del f
        torch.cuda.empty_cache()
        tf.keras.backend.clear_session()

    # Return the rmse and cliff rmse for all folds
    return rmse_scores, cliff_rmse_scores


def smi_tokenizer(smi: str):
    """ Tokenize a SMILES """
    pattern = "(\[|\]|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Si|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    return tokens


def augment(smiles: List[str], *args, augment_factor: int = 10, max_smiles_len: int = 200, max_tries: int = 1000):
    """ Augment SMILES strings by adding non-canonical SMILES. Keeps corresponding activity values/CHEMBL IDs, etc """
    augmented_smiles = []
    augmented_args = [[] for _ in args]
    for i, smi in enumerate(tqdm(smiles)):
        generated = smile_augmentation(smi, augment_factor - 1, max_smiles_len, max_tries)
        augmented_smiles.append(smi)
        augmented_smiles.extend(generated)

        for a, arg in enumerate(args):
            for _ in range(len(generated) + 1):
                augmented_args[a].append(arg[i])

    return tuple([augmented_smiles], ) + tuple(augmented_args)


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


def smile_augmentation(smile: str, augmentation: int, max_len: int = 200, max_tries: int = 1000):
    """Generate n random non-canonical SMILES strings from a SMILES string with length constraints"""
    # https://github.com/michael1788/virtual_libraries/blob/master/experiments/do_data_processing.py
    mol = Chem.MolFromSmiles(smile)
    s = set()
    for i in range(max_tries):
        if len(s) == augmentation:
            break

        smiles = random_smiles(mol)
        if len(smiles) <= max_len:
            tokens = smi_tokenizer(smiles)
            if all([tok in smiles_encoding['token_indices'] for tok in tokens]):
                s.add(smiles)

    return list(s)
