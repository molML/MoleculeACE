"""
Author: Derek van Tilborg -- TU/e -- 20-06-2022

Script for estimating the difference between predictions on the train vs the test cliffs

"""

from MoleculeACE.models import RF, SVM, GBM, KNN, MLP, GCN, MPNN, AFP, GAT, CNN, LSTM, Transformer
from MoleculeACE.benchmark.const import Descriptors, CONFIG_PATH, datasets, WORKING_DIR, RANDOM_SEED
from MoleculeACE.benchmark.utils import Data, get_config, calc_rmse, calc_cliff_rmse, cross_validate, load_model
from tqdm import tqdm
import warnings
import pickle
import os
import pandas as pd
import numpy as np


combinations = {Descriptors.ECFP: [RF, SVM, GBM, KNN, MLP],
                Descriptors.MACCS: [RF, SVM, GBM, KNN],
                Descriptors.PHYSCHEM: [RF, SVM, GBM, KNN],
                Descriptors.WHIM: [RF, SVM, GBM, KNN],
                Descriptors.GRAPH: [GCN, MPNN, AFP, GAT],
                Descriptors.TOKENS: [Transformer],
                Descriptors.SMILES: [CNN, LSTM]}


def write_results(filename, dataset, algo, descriptor, train_rmse, train_cliff_rmse, test_rmse, test_rmse_cliff):

    # Create output file if it doesn't exist already
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.write('dataset,algorithm,descriptor,train_rmse,train_cliff_rmse,test_rmse,test_rmse_cliff\n')
    with open(filename, 'a') as f:
        f.write(f'{dataset},{algo},{descriptor},{augmentation},{train_rmse},{train_cliff_rmse},{test_rmse},'
                f'{test_rmse_cliff}\n')


def load_cross_validate(filename: Union[str, List[str]], data: Data):
    """ Load multiple models and evaluate them """

    if type(filename) is str:
        filename = [filename]

    train_rmse, train_cliff_rmse, test_rmse, test_cliff_rmse = [], [], [], []
    for model_path in filename:
        model = load_model(model_path)

        # Train performance
        y_hat = model.predict(data.x_train)
        train_rmse.append(calc_rmse(data.y_train, y_hat))
        train_cliff_rmse.append(calc_cliff_rmse(y_test_pred=y_hat, y_test=data.y_train,
                                                cliff_mols_test=data.cliff_mols_train))

        # Test performance
        y_hat = model.predict(data.x_test)
        test_rmse.append(calc_rmse(data.y_test, y_hat))
        test_cliff_rmse.append(calc_cliff_rmse(y_test_pred=y_hat, y_test=data.y_test,
                                               cliff_mols_test=data.cliff_mols_test))

    return train_rmse, train_cliff_rmse, test_rmse, test_cliff_rmse


def main(results_filename: str = "train_test_similarity.csv",
         pretrained_models_location: str = os.path.join(WORKING_DIR, 'pretrained_models')):

    # Loop through datasets
    for dataset in tqdm(datasets):
        for descriptor, algorithms in combinations.items():

            # Featurize SMILES strings with a specific method
            data = Data(dataset)
            data(descriptor)

            for algo in algorithms:

                combi = f"{algo.__name__}_{descriptor.name}"

                model_paths = []
                for file in os.path.listdir(os.path.join(pretrained_models_location, dataset)):
                    if file.startswith(combi):
                        model_paths.append(os.path.join(pretrained_models_location, dataset, file))

                # Calculate performance on train and test for all fold models
                train_rmse, train_cliff_rmse, test_rmse, test_cliff_rmse = load_cross_validate(model_paths, data)

                write_results(results_filename, dataset, algo, descriptor,  np.mean(train_rmse),
                              np.mean(train_cliff_rmse), np.mean(test_rmse), np.mean(test_rmse_cliff))


if __name__ == '__main__':
    main("train_test_similarity.csv", os.path.join(WORKING_DIR, 'pretrained_models'))
