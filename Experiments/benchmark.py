"""
Author: Derek van Tilborg -- TU/e -- 09-06-2022

Script to replicate the benchmark results

"""

from MoleculeACE.models import RF, SVM, GBM, KNN, MLP, GCN, MPNN, AFP, GAT, CNN, LSTM, Transformer
from MoleculeACE.benchmark.const import Descriptors, CONFIG_PATH, datasets, WORKING_DIR, RANDOM_SEED
from MoleculeACE.benchmark.utils import Data, get_config, calc_rmse, calc_cliff_rmse, cross_validate
from tqdm import tqdm
import warnings
import pickle
import os
import pandas as pd

combinations = {Descriptors.ECFP: [RF, SVM, GBM, KNN, MLP],
                Descriptors.MACCS: [RF, SVM, GBM, KNN],
                Descriptors.PHYSCHEM: [RF, SVM, GBM, KNN],
                Descriptors.WHIM: [RF, SVM, GBM, KNN],
                Descriptors.GRAPH: [GCN, MPNN, AFP, GAT],
                Descriptors.TOKENS: [Transformer],
                Descriptors.SMILES: [CNN, LSTM]}

# How much augmentation we use on SMILES based methods (CNN, LSTM, Transformer)
AUGMENTATION_FACTOR = 10


def write_results(filename, dataset, algo, descriptor, augmentation, rmse, cliff_rmse, data):

    # Create output file if it doesn't exist already
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.write('dataset, algorithm, descriptor, augmentation, rmse, cliff_rmse, n_compounds, n_cliff_compounds,'
                    'n_compounds_train, n_cliff_compounds_train, n_compounds_test, n_cliff_compounds_test\n')
    with open(filename, 'a') as f:
        f.write(f'{dataset}, {algo}, {descriptor}, {augmentation}, {rmse}, {cliff_rmse}, '
                f'{len(data.y_train) + len(data.y_test)}, {sum(data.cliff_mols_train) + sum(data.cliff_mols_test)},'
                f'{len(data.y_train)}, {sum(data.cliff_mols_train)}, {len(data.y_test)}, {sum(data.cliff_mols_test)}\n')


def main(results_filename: str = "MoleculeACE_results.csv"):

    # Loop through datasets
    for dataset in tqdm(datasets):

        if dataset not in pd.read_csv(results_filename)['dataset'].tolist():

            data = Data(dataset)

            for descriptor, algorithms in combinations.items():

                if descriptor in [Descriptors.SMILES, Descriptors.TOKENS]:
                    data.augment(AUGMENTATION_FACTOR)
                    data.shuffle()

                # Featurize SMILES strings with a specific method
                data(descriptor)

                for algo in algorithms:

                    combi = f"{algo.__name__}_{descriptor.name}"
                    config_path = os.path.join(CONFIG_PATH, 'benchmark', dataset, f"{combi}.yml")
                    model_save_path = os.path.join(WORKING_DIR, 'pretrained_models', dataset, f"{combi}.pkl")

                    try:

                        # Get the optimized hyperparameters from a config file
                        hyperparameters = get_config(config_path)
                        if "epochs" in hyperparameters:
                            hyperparameters.pop("epochs")

                        # Train model
                        if algo in [RF, SVM, GBM, KNN]:
                            f = algo(**hyperparameters)
                            f.train(data.x_train, data.y_train)

                            with open(model_save_path, 'wb') as handle:
                                pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)

                            y_hat = f.predict(data.x_test)

                            rmse = calc_rmse(data.y_test, y_hat)
                            rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=data.y_test,
                                                         cliff_mols_test=data.cliff_mols_test)

                        else:
                            rmse, rmse_cliff = cross_validate(algo, data, n_folds=5, early_stopping=10, seed=RANDOM_SEED,
                                                              save_path=model_save_path, **hyperparameters)

                            # Get the fold average
                            rmse = sum(rmse)/len(rmse)
                            rmse_cliff = sum(rmse_cliff)/len(rmse_cliff)

                        write_results(filename=results_filename, dataset=dataset,
                                      algo=algo.__name__, descriptor=descriptor.name,
                                      augmentation=data.augmented,
                                      rmse=rmse, cliff_rmse=rmse_cliff, data=data)

                    except:
                        warnings.warn(f" -- FAILED {dataset}-{combi}")


if __name__ == '__main__':
    main(results_filename="MoleculeACE_results.csv")
