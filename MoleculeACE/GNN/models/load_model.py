"""
Load a GNN model
Alisa Alenicheva, Jetbrains research, Februari 2022
"""

import torch

from MoleculeACE.GNN.models import init_model
from MoleculeACE.benchmark import utils


def load_model(algorithm, descriptor, dataset, model_file, config_file):
    config = utils.get_config(config_file)
    model = init_model(config, algorithm, descriptor)
    model.load_state_dict(torch.load(model_file))
    return model
