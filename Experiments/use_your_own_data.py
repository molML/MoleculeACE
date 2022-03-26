"""
With this script you can train one of the 23 included models with your own data

load one of the pre-processed and split datasets that comes with MoleculeACE
You can find all included datasets here: utils.datasets
Supported descriptors and ML algorithms can be found here: benchmark.Descriptors, benchmark.Algorithms
"""

from MoleculeACE.benchmark import load_data, models, evaluation, utils, process_data
import os

# Setup some variables
dataset = os.path.join(utils.WORKING_DIR, 'benchmark_data', 'CHEMBL287_Ki.csv')
descriptor = utils.Descriptors.ECFP
algorithm = utils.Algorithms.GBM

if __name__ == '__main__':

    # Process your data data
    process_data(dataset, smiles_colname='smiles', y_colname='exp_mean [nM]', test_size=0.2,
                 fold_threshold=10, similarity_threshold=0.9)

    # Load data
    data = load_data(dataset, descriptor=descriptor, tolog10=True)

    # Train and optimize a model. You can also implement your own model here
    model = models.train_model(data, algorithm=algorithm, config_file=None)
    predictions = model.test_predict()

    # Evaluate your model on activity cliff compounds
    results = evaluation.evaluate(data=data, predictions=predictions)
