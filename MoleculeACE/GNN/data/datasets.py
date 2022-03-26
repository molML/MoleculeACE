"""
Datasets for the GNN models
Alisa Alenicheva, Jetbrains research, Februari 2022
"""

import random
from functools import partial

import numpy as np
import pandas as pd
import torch
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer, \
    AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, PretrainAtomFeaturizer, PretrainBondFeaturizer

from MoleculeACE.benchmark.utils import RANDOM_SEED, Descriptors, get_config
from MoleculeACE.benchmark.utils.const import DATA_PATH, CONFIG_PATH_GENERAL
general_settings= get_config(CONFIG_PATH_GENERAL)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def get_moleculecsv_dataset(smiles, activity, descriptor):
    df = pd.DataFrame(columns=["smiles", "log10"])
    df["smiles"] = smiles
    df["log10"] = activity
    if descriptor == Descriptors.CANONICAL_GRAPH:
        node_featurizer = CanonicalAtomFeaturizer()
        edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    elif descriptor == Descriptors.ATTENTIVE_GRAPH:
        node_featurizer = AttentiveFPAtomFeaturizer()
        edge_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
    elif descriptor == Descriptors.PRETRAINED_GRAPH:
        node_featurizer = PretrainAtomFeaturizer()
        edge_featurizer = PretrainBondFeaturizer()
    else:
        raise ValueError('Unexpected descriptor: {}'.format(descriptor.value))
    dataset = MoleculeCSVDataset(df=df,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                 node_featurizer=node_featurizer,
                                 edge_featurizer=edge_featurizer,
                                 smiles_column="smiles",
                                 task_names=['log10'],
                                 log_every=100,
                                 cache_file_path=f"{DATA_PATH}/data_processed",
                                 load=False,
                                 n_jobs=general_settings['num_workers'])

    return dataset
