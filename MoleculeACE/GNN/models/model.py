"""
Functions to initialize GNNs
Alisa Alenicheva, Jetbrains research, Februari 2022
"""

import torch
from dgllife.model import GINPredictor, load_pretrained
from torch.nn import functional as F

from MoleculeACE.GNN.models.utils import get_atom_feat_size, get_bond_feat_size
from MoleculeACE.benchmark.utils import Algorithms, RANDOM_SEED, Descriptors
import random
import numpy as np

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def GCN_model(node_feature_size, config):
    from dgllife.model import GCNPredictor
    model = GCNPredictor(
        in_feats=node_feature_size,
        hidden_feats=[config['gnn_hidden_feats']] * config['num_gnn_layers'],
        activation=[F.relu] * config['num_gnn_layers'],
        residual=[config['residual']] * config['num_gnn_layers'],
        batchnorm=[config['batchnorm']] * config['num_gnn_layers'],
        dropout=[config['dropout']] * config['num_gnn_layers'],
        predictor_hidden_feats=config['predictor_hidden_feats'],
        predictor_dropout=config['dropout'],
        n_tasks=1)
    return model


def MPNN_model(node_feature_size, bond_feature_size, config):
    from dgllife.model import MPNNPredictor
    model = MPNNPredictor(
        node_in_feats=node_feature_size,
        edge_in_feats=bond_feature_size,
        node_out_feats=config['node_out_feats'],
        edge_hidden_feats=config['edge_hidden_feats'],
        num_step_message_passing=config['num_step_message_passing'],
        num_step_set2set=config['num_step_set2set'],
        num_layer_set2set=config['num_layer_set2set'],
        n_tasks=1)
    return model


def GAT_model(node_feature_size, config):
    from dgllife.model import GATPredictor
    model = GATPredictor(
        in_feats=node_feature_size,
        hidden_feats=[config['gnn_hidden_feats']] * config['num_gnn_layers'],
        num_heads=[config['num_heads']] * config['num_gnn_layers'],
        feat_drops=[config['dropout']] * config['num_gnn_layers'],
        attn_drops=[config['dropout']] * config['num_gnn_layers'],
        alphas=[config['alpha']] * config['num_gnn_layers'],
        residuals=[config['residual']] * config['num_gnn_layers'],
        predictor_hidden_feats=config['predictor_hidden_feats'],
        predictor_dropout=config['dropout'],
        n_tasks=1)
    return model


def AttentiveFP_model(node_feature_size, bond_feature_size, config):
    from dgllife.model import AttentiveFPPredictor
    model = AttentiveFPPredictor(
        node_feat_size=node_feature_size,
        edge_feat_size=bond_feature_size,
        num_layers=config['num_layers'],
        num_timesteps=config['num_timesteps'],
        graph_feat_size=config['graph_feat_size'],
        dropout=config['dropout'],
        n_tasks=1)
    return model

def GIN_model(algorithm, config):
    model = GINPredictor(
        num_node_emb_list=[120, 3],
        num_edge_emb_list=[6, 3],
        num_layers=5,
        emb_dim=300,
        JK=config['jk'],
        dropout=0.5,
        readout=config['readout'],
        n_tasks=1
    )
    model.gnn = load_pretrained(algorithm.value)
    model.gnn.JK = config['jk']
    return model


def init_model(config, algorithm: Algorithms, descriptor: Descriptors):
    node_feature_size = get_atom_feat_size(descriptor)
    if algorithm == Algorithms.GCN:
        model = GCN_model(node_feature_size, config)
    elif algorithm == Algorithms.GAT:
        model = GAT_model(node_feature_size, config)
    elif algorithm == Algorithms.MPNN or algorithm == Algorithms.AFP:
        bond_feature_size = get_bond_feat_size(descriptor)
        if algorithm == Algorithms.AFP:
            model = AttentiveFP_model(node_feature_size, bond_feature_size, config)
        else:
            model = MPNN_model(node_feature_size, bond_feature_size, config)
    elif algorithm in [Algorithms.GIN_CONTEXTPRED, Algorithms.GIN_EDGEPRED, Algorithms.GIN_INFOMAX, Algorithms.GIN_MASKING]:
        model = GIN_model(algorithm, config)
    else:
        raise ValueError('Unexpected model: {}'.format(algorithm.value))
    return model
