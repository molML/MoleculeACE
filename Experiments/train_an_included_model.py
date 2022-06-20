"""
With this script you can train one of the 24 included models

load one of the pre-processed datasets that comes with MoleculeACE
"""

from MoleculeACE.models import RF, SVM, GBM, KNN, MLP, GCN, MPNN, AFP, GAT, CNN, LSTM, Transformer
from MoleculeACE.benchmark.const import Descriptors, CONFIG_PATH, datasets, WORKING_DIR, RANDOM_SEED
from MoleculeACE.benchmark.utils import Data, get_config, calc_rmse, calc_cliff_rmse, cross_validate
import os


# Define the dataset, descriptor, and algorithm.
dataset = 'CHEMBL2034_Ki'  # All available datasets are found at MoleculeACE.benchmark.const.datasets
descriptor = Descriptors.GRAPH  # pick between Descriptors.ECFP, MACCS, PHYSCHEM, WHIM, Graph, SMILES, TOKENS
algorithm = MPNN  # pick between RF, SVM, GBM, KNN, MLP, GCN, MPNN, AFP, GAT, CNN, LSTM, Transformer

config_path = os.path.join(CONFIG_PATH, 'benchmark', dataset, f"{algorithm.__name__}_{descriptor.name}.yml")
hyperparameters = get_config(config_path)


def main():

    # Load data
    data = Data(dataset)

    # Featurize SMILES strings with a specific method
    data(descriptor)

    # Get the best hyperparameters from a config file (all datasets+models have already been optimized)
    config_path = os.path.join(CONFIG_PATH, 'benchmark', dataset, f"{algorithm.__name__}_{descriptor.name}.yml")
    hyperparameters = get_config(config_path)

    # Train and a model
    model = algorithm(**hyperparameters)
    model.train(data.x_train, data.y_train)
    y_hat = model.predict(data.x_test)

    # Evaluate your model on activity cliff compounds
    rmse = calc_rmse(data.y_test, y_hat)
    rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=data.y_test, cliff_mols_test=data.cliff_mols_test)

    print(f"rmse: {rmse}\nrmse_cliff: {rmse_cliff}")


if __name__ == '__main__':
    main()
