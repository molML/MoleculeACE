"""
Function to train a model
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

import os

from MoleculeACE.benchmark.utils.const import Algorithms, Descriptors, WORKING_DIR
from .model import Model
from .. import Data


def train_model(data: Data, algorithm=Algorithms.RF, config_file: str = None,
                pretrained_model=os.path.join(WORKING_DIR, 'pretrain_model', 'lstm_next_token.h5')):
    """ Train a machine learning model

    Args:
        pretrained_model: (str) Path to pre-trained model. Only the LSTM currently supports this
        config_file: (str) Path to config file. If None, optimize hyperparameters and create general config file, if
        the file doesn't exist yet, optimize hyperparameters and save best config as this path.
        data: MoleculeACE.benchmark.Data object containing train x and y data
        algorithm: (str) algorithm to use

    Returns: MoleculeACE.benchmark.Model

    """
    if algorithm in [Algorithms.GIN_CONTEXTPRED, Algorithms.GIN_INFOMAX, Algorithms.GIN_EDGEPRED,
                     Algorithms.GIN_MASKING]:
        if data.descriptor != Descriptors.PRETRAINED_GRAPH:
            print(f"For model {algorithm.value} only {Descriptors.PRETRAINED_GRAPH.value} descriptors are available \n"
                  f"{data.descriptor.value} will be left and {Descriptors.PRETRAINED_GRAPH.value} will be applied")
            data.descriptor = Descriptors.PRETRAINED_GRAPH
    model = Model(data, algorithm=algorithm, descriptor=data.descriptor)
    model.train(algorithm=algorithm, cv=data.get_cv_folds(), pretrained_model=pretrained_model,
                config_file=config_file)

    return model
