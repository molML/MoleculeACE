
"""
Author: Derek van Tilborg -- TU/e -- 09-06-2022

Script for bayesian optimization of model hyperparameters using 5-fold cross-validation.

"""

from MoleculeACE.models import RF, SVM, GBM, KNN, MLP, GCN, MPNN, AFP, GAT, CNN
from MoleculeACE.models.optimize import BayesianOptimization
from MoleculeACE.benchmark.const import Descriptors, CONFIG_PATH, datasets, WORKING_DIR
from MoleculeACE.benchmark.utils import Data, get_config
from tqdm import tqdm
import warnings
import os

AUGMENT = 10  # augment SMILES strings n times (only used for sequence models)
N_CALLS = 50  # n optimization attempts

combinations = {Descriptors.ECFP: [RF, SVM, GBM, KNN, MLP],
                Descriptors.MACCS: [RF, SVM, GBM, KNN],
                Descriptors.PHYSCHEM: [RF, SVM, GBM, KNN],
                Descriptors.WHIM: [RF, SVM, GBM, KNN],
                Descriptors.GRAPH: [GCN, MPNN, AFP, GAT],
                Descriptors.SMILES: [CNN]}  # the LSTM and Transformer are not systematically optimized


def main():

    for dataset in tqdm(datasets):

        data = Data(dataset)

        # If we're using sequence models on SMILES strings directly, use data augmentation
        for descriptor, algorithms in combinations.items():
            if descriptor in [Descriptors.SMILES, Descriptors.TOKENS]:
                data.augment(AUGMENT)
                data.shuffle()

            # Featurize SMILES strings with a specific method
            data(descriptor)

            for algo in algorithms:

                combi = f"{algo.__name__}_{descriptor.name}"

                try:
                    # Get the default config
                    config = get_config(os.path.join(CONFIG_PATH, 'default', f"{algo.__name__}.yml"))

                    # Perform Bayesian optimization
                    opt = BayesianOptimization(algo)
                    opt.optimize(data.x_train, data.y_train, config, n_calls=N_CALLS)

                    # Save best hyperparameters as a config file
                    opt.save_config(os.path.join(CONFIG_PATH, 'benchmark', dataset, f"{combi}.yml"))
                except:
                    warnings.warn(f" -- FAILED {dataset}-{combi}")


if __name__ == '__main__':
    main()
