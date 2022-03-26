"""
With this script you can load an included model

load one of the pre-processed and split datasets that comes with MoleculeACE
You can find all included datasets here: utils.datasets
Supported descriptors and ML algorithms can be found here: utils.Descriptors, utils.Algorithms
"""

from MoleculeACE.benchmark import load_data, models, evaluation, utils

# Setup some variables
dataset = 'CHEMBL287_Ki'
descriptor = utils.Descriptors.ATTENTIVE_GRAPH
algorithm = utils.Algorithms.AFP
path_to_model = 'path_to_model.pkl'

if __name__ == '__main__':

    # Load data
    data = load_data(dataset, descriptor=descriptor)

    # Load a model
    model = models.load_model(data, algorithm, path_to_model)
    predictions = model.test_predict()

    # Evaluate your model on activity cliff compounds
    results = evaluation.evaluate(data=data, predictions=predictions)
