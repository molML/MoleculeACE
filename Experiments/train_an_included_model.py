"""
With this script you can train one of the 23 included models

load one of the pre-processed and split datasets that comes with MoleculeACE
You can find all included datasets here: utils.datasets
Supported descriptors and ML algorithms can be found here: utils.Descriptors, utils.Algorithms
"""

from MoleculeACE.benchmark import load_data, models, evaluation, utils
import os

# Setup some variables
dataset = 'CHEMBL287_Ki'
descriptor = utils.Descriptors.ECFP
algorithm = utils.Algorithms.GBM
config_file = os.path.join(utils.WORKING_DIR, 'configures', 'benchmark', dataset,
                           f'{algorithm.value}_{descriptor.value}.yml')

if __name__ == '__main__':

    # Load data
    data = load_data(dataset, descriptor=descriptor)

    # Train and a model, if config_file = None, hyperparameter optimization is performed
    model = models.train_model(data, algorithm=algorithm, config_file=config_file)
    predictions = model.test_predict()

    # Evaluate your model on activity cliff compounds
    results = evaluation.evaluate(data=data, predictions=predictions)
