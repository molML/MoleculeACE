"""
Bayesian optimization function for hyperparameter optimization
Alisa Alenicheva, Jetbrains research, Februari 2022
"""

import json
import os
import random
import numpy as np

import torch
from dgllife.utils import ScaffoldSplitter
from hyperopt import fmin, tpe

from MoleculeACE.GNN.data.dataloaders import get_train_val_dataloaders
from MoleculeACE.GNN.models.train_test_val import train_pipeline
from MoleculeACE.benchmark.utils import Algorithms
from MoleculeACE.benchmark.utils.const import RANDOM_SEED, Descriptors, define_default_log_dir, \
    CONFIG_PATH_GENERAL

from MoleculeACE.benchmark.utils import get_config
general_settings = get_config(CONFIG_PATH_GENERAL)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def bayesian_optimization(dataset, algorithm: Algorithms, descriptor: Descriptors,
                          n_folds=general_settings['num_cv_folds'],
                          num_evals=general_settings['bayesian_optimization_tries'],
                          result_path='', logs_path=None):
    if logs_path is None:
        logs_path = define_default_log_dir()
    from MoleculeACE.GNN.models.optimization.optimization_util import init_hyper_space, init_trial_path

    # Run grid search
    results = []
    candidate_hypers = init_hyper_space(algorithm)

    def objective(hyperparams):
        trial_path = init_trial_path(result_path)

        k_fold_split = ScaffoldSplitter.k_fold_split(dataset, k=n_folds)

        val_metric = 0
        for fold_num, (train_set, val_set) in enumerate(k_fold_split):
            train_loader, val_loader = get_train_val_dataloaders(train_set, val_set, hyperparams['batch_size'])

            # Initialize the model
            model, stopper = train_pipeline(train_loader, val_loader, descriptor, algorithm, hyperparams['epochs'],
                                            hyperparams, result_path, logs_path)

            val_metric += stopper.best_score

        results.append((trial_path, val_metric / n_folds, hyperparams))
        with open(os.path.join(trial_path, 'configure.json'), 'w') as f:
            json.dump(hyperparams, f, indent=2)
        with open(os.path.join(trial_path, 'eval.txt'), 'w') as f:
            f.write('Best val {}: {}\n'.format(general_settings['metric'], stopper.best_score))

        return val_metric / n_folds

    fmin(objective, candidate_hypers, algo=tpe.suggest, max_evals=num_evals)
    results.sort(key=lambda tup: tup[1])
    best_trial_path, best_val_metric, hypers = results[0]

    return best_trial_path, best_val_metric, hypers
