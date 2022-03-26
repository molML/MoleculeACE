"""
Load a model
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

from MoleculeACE.benchmark.utils.const import Algorithms
from .model import Model


def load_model(data, algorithm: Algorithms, model_file: str):
    """ Train a machine learning model

    Args:
        data: MoleculeACE.benchmark.Data object containing train x and y data
        algorithm: MoleculeACE.benchmark.utils.Algorithms object - algorithm to use
        model_path: string path to file with model. All models use pickle files except CNN, MLP, and LSTM which use .h5

    Returns: MoleculeACE.benchmark.Model

    """
    model = Model(data, algorithm=algorithm, descriptor=data.descriptor)
    model.load_model(model_file)

    return model