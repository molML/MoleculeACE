"""
Dataloader for the GNN models
Alisa Alenicheva, Jetbrains research, Februari 2022
"""

from torch.utils.data import DataLoader
from MoleculeACE.benchmark.utils import collate_molgraphs, RANDOM_SEED
import torch
import random
import numpy as np

from MoleculeACE.benchmark.utils.const import CONFIG_PATH_GENERAL
from MoleculeACE.benchmark.utils import get_config
general_settings = get_config(CONFIG_PATH_GENERAL)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


def get_train_val_test_dataloaders(train_set, val_set, test_set, batch_size,
                                   num_workers=general_settings['num_workers']):
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=num_workers,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_molgraphs, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_molgraphs, num_workers=num_workers,
                             worker_init_fn=seed_worker, generator=g)
    return train_loader, val_loader, test_loader


def get_train_val_dataloaders(train_set, val_set, batch_size, num_workers=general_settings['num_workers']):
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=num_workers,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_molgraphs, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=g)
    return train_loader, val_loader


def split_dataset_in_loaders(train_set, validation_split, batch_size, num_workers=general_settings['num_workers']):
    from dgllife.utils import ScaffoldSplitter

    # Split the train and validation set by molecular scaffolds
    train_set, val_set, _ = ScaffoldSplitter.train_val_test_split(train_set, mols=None, sanitize=True,
                                                                  frac_train=1 - validation_split,
                                                                  frac_val=validation_split,
                                                                  frac_test=0, log_every_n=1000,
                                                                  scaffold_func='decompose')

    # Get the train and validation dataloaders
    train_loader, val_loader = get_train_val_dataloaders(train_set, val_set,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers)

    return train_loader, val_loader
