"""
Some random functions for hyperparameter optimization
Alisa Alenicheva, Jetbrains research, Februari 2022
"""

import os

import torch
from hyperopt import hp
import errno
from MoleculeACE.benchmark.utils import get_config
from MoleculeACE.benchmark.utils.const import Algorithms, RANDOM_SEED, CONFIG_PATH, CONFIG_PATH_GIN, CONFIG_PATH_MPNN, \
    CONFIG_PATH_GCN, CONFIG_PATH_AFP, CONFIG_PATH_GAT

import random
import numpy as np
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


cf_gcn = get_config(CONFIG_PATH_GCN)
gcn_hyperparameters = {
    'epochs': hp.choice('epochs', [cf_gcn['epochs']]),
    'val_split': hp.choice('val_split', [cf_gcn['val_split']]),
    'lr': hp.uniform('lr', low=cf_gcn['lr_min'], high=cf_gcn['lr_max']),
    'weight_decay': hp.uniform('weight_decay', low=cf_gcn['weight_decay_min'], high=cf_gcn['weight_decay_max']),
    'patience': hp.choice('patience', [cf_gcn['early_stopping_patience']]),
    'batch_size': hp.choice('batch_size', cf_gcn['batch_size']),
    'gnn_hidden_feats': hp.choice('gnn_hidden_feats', cf_gcn['gnn_hidden_feats']),
    'predictor_hidden_feats': hp.choice('predictor_hidden_feats', cf_gcn['predictor_hidden_feats']),
    'num_gnn_layers': hp.choice('num_gnn_layers', cf_gcn['num_gnn_layers']),
    'residual': hp.choice('residual', cf_gcn['residual']),
    'batchnorm': hp.choice('batchnorm', cf_gcn['batchnorm']),
    'dropout': hp.uniform('dropout', low=cf_gcn['dropout'][0], high=cf_gcn['dropout'][1])
}

cf_gat = get_config(CONFIG_PATH_GAT)
gat_hyperparameters = {
    'epochs': hp.choice('epochs', [cf_gat['epochs']]),
    'val_split': hp.choice('val_split', [cf_gat['val_split']]),
    'lr': hp.uniform('lr', low=cf_gat['lr_min'], high=cf_gat['lr_max']),
    'weight_decay': hp.uniform('weight_decay', low=cf_gat['weight_decay_min'], high=cf_gat['weight_decay_max']),
    'patience': hp.choice('patience', [cf_gat['early_stopping_patience']]),
    'batch_size': hp.choice('batch_size', cf_gat['batch_size']),
    'gnn_hidden_feats': hp.choice('gnn_hidden_feats', cf_gat['gnn_hidden_feats']),
    'num_heads': hp.choice('num_heads', cf_gat['num_heads']),
    'alpha': hp.uniform('alpha', low=cf_gat['alpha'][0], high=cf_gat['alpha'][1]),
    'predictor_hidden_feats': hp.choice('predictor_hidden_feats', cf_gat['predictor_hidden_feats']),
    'num_gnn_layers': hp.choice('num_gnn_layers', cf_gat['num_gnn_layers']),
    'residual': hp.choice('residual', cf_gat['residual']),
    'dropout': hp.uniform('dropout', low=cf_gat['dropout'][0], high=cf_gat['dropout'][1])
}


cf_mpnn = get_config(CONFIG_PATH_MPNN)
mpnn_hyperparameters = {
    'epochs': hp.choice('epochs', [cf_mpnn['epochs']]),
    'val_split': hp.choice('val_split', [cf_mpnn['val_split']]),
    'lr': hp.uniform('lr', low=cf_mpnn['lr_min'], high=cf_mpnn['lr_max']),
    'weight_decay': hp.uniform('weight_decay', low=cf_mpnn['weight_decay_min'], high=cf_mpnn['weight_decay_max']),
    'patience': hp.choice('patience', [cf_mpnn['early_stopping_patience']]),
    'batch_size': hp.choice('batch_size', cf_mpnn['batch_size']),
    'node_out_feats': hp.choice('node_out_feats', cf_mpnn['node_out_feats']),
    'edge_hidden_feats': hp.choice('edge_hidden_feats', cf_mpnn['edge_hidden_feats']),
    'num_step_message_passing': hp.choice('num_step_message_passing', cf_mpnn['num_step_message_passing']),
    'num_step_set2set': hp.choice('num_step_set2set', cf_mpnn['num_step_set2set']),
    'num_layer_set2set': hp.choice('num_layer_set2set', cf_mpnn['num_layer_set2set'])
}


cf_afp = get_config(CONFIG_PATH_AFP)
attentivefp_hyperparameters = {
    'epochs': hp.choice('epochs', [cf_afp['epochs']]),
    'val_split': hp.choice('val_split', [cf_afp['val_split']]),
    'lr': hp.uniform('lr', low=cf_afp['lr_min'], high=cf_afp['lr_max']),
    'weight_decay': hp.uniform('weight_decay', low=cf_afp['weight_decay_min'], high=cf_afp['weight_decay_max']),
    'patience': hp.choice('patience', [cf_afp['early_stopping_patience']]),
    'batch_size': hp.choice('batch_size', cf_afp['batch_size']),
    'num_layers': hp.choice('num_layers', cf_afp['num_layers']),
    'num_timesteps': hp.choice('num_timesteps', cf_afp['num_timesteps']),
    'graph_feat_size': hp.choice('graph_feat_size', cf_afp['graph_feat_size']),
    'dropout': hp.uniform('dropout', low=cf_afp['dropout'][0], high=cf_afp['dropout'][1])
}


cf_gin = get_config(CONFIG_PATH_GIN)
gin_pretrained_hyperparameters = {
    'epochs': hp.choice('epochs', [cf_gin['epochs']]),
    'val_split': hp.choice('val_split', [cf_gin['val_split']]),
    'lr': hp.uniform('lr', low=cf_gin['lr_min'], high=cf_gin['lr_max']),
    'weight_decay': hp.uniform('weight_decay', low=cf_gin['weight_decay_min'], high=cf_gin['weight_decay_max']),
    'patience': hp.choice('patience', [cf_gin['early_stopping_patience']]),
    'batch_size': hp.choice('batch_size', cf_gin['batch_size']),
    'jk': hp.choice('jk', cf_gin['jk']),
    'readout': hp.choice('readout', cf_gin['readout'])
}


def init_hyper_space(model: Algorithms):
    """Initialize the hyperparameter search space
    Parameters
    ----------
    model : str
        Model for searching hyperparameters
    Returns
    -------
    dict
        Mapping hyperparameter names to the associated search spaces
    """
    candidate_hypers = dict()
    if model == Algorithms.GCN:
        candidate_hypers.update(gcn_hyperparameters)
    elif model == Algorithms.GAT:
        candidate_hypers.update(gat_hyperparameters)
    elif model == Algorithms.MPNN:
        candidate_hypers.update(mpnn_hyperparameters)
    elif model == Algorithms.AFP:
        candidate_hypers.update(attentivefp_hyperparameters)
    elif model in [Algorithms.GIN_MASKING, Algorithms.GIN_INFOMAX, Algorithms.GIN_EDGEPRED, Algorithms.GIN_CONTEXTPRED]:
        candidate_hypers.update(gin_pretrained_hyperparameters)
    else:
        return ValueError('Unexpected model: {}'.format(model))
    return candidate_hypers


def mkdir_p(path):
    """Create a folder for the given path.
    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise


def init_trial_path(result_path):
    """ Check and get the trial output path

    Args:
        result_path: (str) Path to save results

    Returns: (str) path to save a optimization trial

    """
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = os.path.join(result_path,
                                       str(trial_id))  # args['result_path'] + '/{:d}'.format(trial_id)
        path_exists = os.path.exists(path_to_results)
    trial_path = path_to_results
    mkdir_p(trial_path)

    return trial_path
