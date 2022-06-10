"""
With this script you can train one of the 23 included models

load one of the pre-processed and split datasets that comes with MoleculeACE
You can find all included datasets here: utils.datasets
Supported descriptors and ML algorithms can be found here: utils.Descriptors, utils.Algorithms
"""

from MoleculeACE.models import RF, SVM, GBM, KNN, MLP, GCN, MPNN, AFP, GAT, CNN, LSTM, Transformer
from MoleculeACE.benchmark.const import Descriptors, CONFIG_PATH, datasets, WORKING_DIR, RANDOM_SEED
from MoleculeACE.benchmark.utils import Data, get_config, calc_rmse, calc_cliff_rmse, cross_validate
import os


# Setup some variables
dataset = 'CHEMBL287_Ki'
descriptor = Descriptors.ECFP
algorithm = GBM

config_path = os.path.join(CONFIG_PATH, 'benchmark', dataset, f"{algorithm.__name__}_{descriptor.name}.yml")
hyperparameters = get_config(config_path)


if __name__ == '__main__':

    # Load data
    data = Data(dataset)

    # Featurize SMILES strings with a specific method
    data(descriptor)

    # Train and a model, if config_file = None, hyperparameter optimization is performed
    model = algorithm(**hyperparameters)
    model.train(data.x_train, data.y_train)

    # Evaluate your model on activity cliff compounds
    y_hat = model.predict(data.x_test)

    rmse = calc_rmse(data.y_test, y_hat)
    rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=data.y_test, cliff_mols_test=data.cliff_mols_test)

    print(f"RMSE: {rmse:.4f}\nRMSEcliff: {rmse_cliff:.4f}")
