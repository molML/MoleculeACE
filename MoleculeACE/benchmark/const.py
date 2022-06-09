"""
This file holds many variables that are used throughout MoleculeACE

Author: Derek van Tilborg -- TU/e -- March 2022

"""

import datetime
import os
from enum import Enum
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

###
# Benchmark parameters
###


RANDOM_SEED = 42


class Algorithms(Enum):
    RF = 'RF'
    GBM = 'GBM'
    GBR = 'GBR'
    SVM = 'SVM'
    SVR = 'SVR'
    KNN = 'KNN'
    MLP = 'MLP'
    CNN = 'CNN'
    LSTM = 'LSTM'
    GCN = 'GCN'
    MPNN = 'MPNN'
    AFP = 'AFP'
    GAT = 'GAT'
    TRANSFORMER = 'TRANS'


class Descriptors(Enum):
    ECFP = 'ECFP'
    MACCS = 'MACCs'
    PHYSCHEM = 'Physchem'
    WHIM = 'WHIM'
    GRAPH = 'Graph'
    SMILES = 'SMILES'
    TOKENS = 'Tokens'


package_path = os.path.join(get_project_root(), 'MoleculeACE')
current_work_dir = os.getcwd()

WORKING_DIR = os.path.join(package_path, "Data")
DATA_PATH = os.path.join(WORKING_DIR, "benchmark_data")
LOGS_PATH = os.path.join(WORKING_DIR, "logs")
CONFIG_PATH = os.path.join(WORKING_DIR, "configures")


###
# All default configures paths
###


CONFIG_PATH_GENERAL = os.path.join(CONFIG_PATH, 'default', 'GENERAL.yml')
CONFIG_PATH_SMILES = os.path.join(CONFIG_PATH, 'default', 'SMILES.yml')

CONFIG_PATH_RF = os.path.join(CONFIG_PATH, 'default', 'RF.yml')
CONFIG_PATH_SVM = os.path.join(CONFIG_PATH, 'default', 'SVM.yml')
CONFIG_PATH_GBM = os.path.join(CONFIG_PATH, 'default', 'GBM.yml')
CONFIG_PATH_KNN = os.path.join(CONFIG_PATH, 'default', 'KNN.yml')
CONFIG_PATH_MLP = os.path.join(CONFIG_PATH, 'default', 'MLP.yml')
CONFIG_PATH_LSTM = os.path.join(CONFIG_PATH, 'default', 'LSTM.yml')
CONFIG_PATH_CNN = os.path.join(CONFIG_PATH, 'default', 'CNN.yml')
CONFIG_PATH_GCN = os.path.join(CONFIG_PATH, 'default', 'GCN.yml')
CONFIG_PATH_GIN = os.path.join(CONFIG_PATH, 'default', 'GIN.yml')
CONFIG_PATH_GNN = os.path.join(CONFIG_PATH, 'default', 'GNN.yml')
CONFIG_PATH_AFP = os.path.join(CONFIG_PATH, 'default', 'AFP.yml')
CONFIG_PATH_GAT = os.path.join(CONFIG_PATH, 'default', 'GAT.yml')
CONFIG_PATH_MPNN = os.path.join(CONFIG_PATH, 'default', 'MPNN.yml')
CONFIG_PATH_TRANS = os.path.join(CONFIG_PATH, 'default', 'TRANSFORMER.yml')


datasets = ['CHEMBL4203_Ki', 'CHEMBL2034_Ki', 'CHEMBL233_Ki', 'CHEMBL4616_EC50', 'CHEMBL287_Ki', 'CHEMBL218_EC50',
            'CHEMBL264_Ki', 'CHEMBL219_Ki', 'CHEMBL2835_Ki', 'CHEMBL2147_Ki', 'CHEMBL231_Ki', 'CHEMBL3979_EC50',
            'CHEMBL237_EC50', 'CHEMBL244_Ki', 'CHEMBL4792_Ki', 'CHEMBL1871_Ki', 'CHEMBL237_Ki', 'CHEMBL262_Ki',
            'CHEMBL2047_EC50', 'CHEMBL239_EC50', 'CHEMBL2971_Ki', 'CHEMBL204_Ki', 'CHEMBL214_Ki', 'CHEMBL1862_Ki',
            'CHEMBL234_Ki', 'CHEMBL238_Ki', 'CHEMBL235_EC50', 'CHEMBL4005_Ki', 'CHEMBL236_Ki', 'CHEMBL228_Ki']


def define_default_log_dir():
    dt = datetime.datetime.now()
    default_log_dir = os.path.join(LOGS_PATH, f"{dt.date()}_{dt.hour}_{dt.minute}_{dt.second}")
    os.makedirs(default_log_dir, exist_ok=True)
    return default_log_dir


def setup_working_dir(path):
    """ Setup a working directory if the user specifies a new one """

    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.exists(os.path.join(path, 'logs')):
        os.mkdir(os.path.join(path, 'logs'))

    if not os.path.exists(os.path.join(path, 'pretrained_models')):
        os.mkdir(os.path.join(path, 'pretrained_models'))

    if not os.path.exists(os.path.join(path, 'benchmark_data')):
        os.mkdir(os.path.join(path, 'benchmark_data'))
        os.mkdir(os.path.join(path, 'benchmark_data', 'train'))
        os.mkdir(os.path.join(path, 'benchmark_data', 'test'))

    if not os.path.exists(os.path.join(path, 'activity_cliffs')):
        os.mkdir(os.path.join(path, 'activity_cliffs'))

    if not os.path.exists(os.path.join(path, 'configures')):
        os.mkdir(os.path.join(path, 'configures'))

    if not os.path.exists(os.path.join(path, 'results')):
        os.mkdir(os.path.join(path, 'results'))
