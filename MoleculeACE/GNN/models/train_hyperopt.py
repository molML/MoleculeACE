"""
Code to train the GNNs
Alisa Alenicheva, Jetbrains research, Februari 2022
"""

import os
import random
from shutil import copyfile

import torch

from MoleculeACE.GNN.data import split_dataset_in_loaders
from MoleculeACE.GNN.data import get_moleculecsv_dataset
from MoleculeACE.GNN.models.optimization import bayesian_optimization
from MoleculeACE.GNN.models.train_test_val import evaluate_model, train_pipeline
from MoleculeACE.benchmark import utils, Data
from MoleculeACE.benchmark.utils import Algorithms, get_config
from MoleculeACE.benchmark.utils.const import RANDOM_SEED, define_default_log_dir, CONFIG_PATH_GENERAL, WORKING_DIR
from MoleculeACE.GNN.models.utils import get_atom_feat_size, get_bond_feat_size
import numpy as np

general_settings= get_config(CONFIG_PATH_GENERAL)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def train_model_with_hyperparameters_optimization(data: Data, config_file, algorithm, descriptor, logs_path=None,
                                                  result_path='', working_dir=WORKING_DIR):
    if logs_path is None:
        logs_path = define_default_log_dir()
    train_set = get_moleculecsv_dataset(data.smiles_train, data.y_train, descriptor)

    # If no valid config file is given, start bayesian optimization and take the best hyperparameters
    if config_file is None or not os.path.exists(config_file):
        # Start bayesian optimization, takes a (long) while
        best_trial_path, best_val_metric, hypers = bayesian_optimization(train_set, algorithm=algorithm,
                                                                         descriptor=descriptor,
                                                                         result_path=logs_path,
                                                                         logs_path=logs_path)
        # Take the best hyperparameters and write them to a config file
        copyfile(best_trial_path + '/configure.json', logs_path + '/configure.json')
        copyfile(best_trial_path + '/eval.txt', logs_path + '/eval.txt')
        config = hypers

        hypers["model"] = algorithm.value
        hypers["n_tasks"] = 1
        hypers["atom_featurizer_type"] = descriptor.value
        hypers["bond_featurizer_type"] = descriptor.value
        hypers["in_node_feats"] = get_atom_feat_size(descriptor)
        if algorithm in [Algorithms.MPNN, Algorithms.AFP]:
            hypers["in_edge_feats"] = get_bond_feat_size(descriptor)

        if config_file is not None:
            if not os.path.exists(config_file):
                utils.write_config(config_file, hypers)
        if config_file is None:

            utils.write_config(os.path.join(working_dir, 'configures', f'{algorithm.value}.yml'), hypers)
    else:
        config = utils.get_config(config_file)

    # Split the train and validation set by molecular scaffolds and put them in train and validation dataloaders
    train_loader, val_loader = split_dataset_in_loaders(train_set, config['val_split'], config['batch_size'],
                                                        num_workers=general_settings['num_workers'])

    # # initiate a GCN model
    model, stopper = train_pipeline(train_loader, val_loader, descriptor, algorithm, config['epochs'], config,
                                    result_path, logs_path)
    # Calculate validation rmse
    val_rmse, val_r2 = evaluate_model(model, val_loader)
    print(f"Validation rmse = {val_rmse}")

    return model
