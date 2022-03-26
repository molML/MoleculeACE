"""
A collection of random functions that are used in different places
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

import json
import os
from yaml import load, Loader, dump


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
    with open(filename, 'w') as file:
        documents = dump(args, file)


def to_list(x):
    """ Put object in a list if it isn't """
    return x if type(x) is list else [x]


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    import dgl
    from torch import stack, ones

    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = ones(labels.shape)
    else:
        masks = stack(masks, dim=0)

    return smiles, bg, labels, masks


def get_torch_device():
    import torch
    """ Get the pytorch device (gpu or cpu)"""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        torch.device('cpu')


def get_atom_features_shape(train_set):
    """Get the shape of the atom features from a MoleculeACE.GNN.dataloaders.*_graph_dataset"""
    atom_features = 74  # this is the standard size
    for item in train_set:
        single_graph = item[1]
        atom_features = single_graph.ndata['h'].shape[1]
        break

    return atom_features


def get_bond_features_shape(train_set):
    """Get the shape of the atom features from a MoleculeACE.GNN.dataloaders.*_graph_dataset"""
    bond_features = 13  # this is the standard size
    for item in train_set:
        single_graph = item[1]
        bond_features = single_graph.edata['e'].shape[1]
        break

    return bond_features


def create_model_checkpoint(save_path):
    """ Function to save the trained model during training """
    from tensorflow.keras.callbacks import ModelCheckpoint

    filepath = os.path.join(save_path, 'best_model.h5')
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True)

    return checkpointer


def plot_history(history, loss='categorical crossentropy loss'):
    import matplotlib.pyplot as plt
    import numpy as np
    from math import ceil as ceiling

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel(loss)
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
             label='Val loss')
    plt.legend()
    plt.ylim([0, ceiling(max(history.history['loss']))])
    plt.show()
