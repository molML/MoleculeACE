""" Model class which can train, test, predict, load, and save models """

import pickle
import os

from MoleculeACE.CNN.models import smiles_cnn
from MoleculeACE.MLP.models import mlp
from MoleculeACE.GNN import train_model_with_hyperparameters_optimization
from MoleculeACE.LSTM.models import lstm, lstm_regression_predict
from MoleculeACE.ML import models as ml
from MoleculeACE.benchmark import Data
from MoleculeACE.benchmark.utils import get_config
from MoleculeACE.benchmark.utils.const import Algorithms, Descriptors, define_default_log_dir, CONFIG_PATH_LSTM, \
    CONFIG_PATH_CNN, CONFIG_PATH_SMILES, WORKING_DIR

import torch

torch.multiprocessing.set_sharing_strategy('file_system')

from tensorflow.keras.models import load_model as keras_load_model

lstm_settings = get_config(CONFIG_PATH_LSTM)
cnn_settings = get_config(CONFIG_PATH_CNN)
smiles_encoding = get_config(CONFIG_PATH_SMILES)


class Model:
    def __init__(self, data: Data, algorithm: Algorithms, descriptor: Descriptors):
        self.data = data
        self.algorithm = algorithm
        self.descriptor = descriptor
        self.x_train = data.x_train
        self.y_train = data.y_train
        self.x_val = data.x_val
        self.y_val = data.y_val
        self.model = None
        self.predictions = None

    def train(self, algorithm: Algorithms, cv: int = 5, config_file: str = None, n_jobs: int = -1,
              pretrained_model: str = os.path.join(WORKING_DIR, 'pretrain_model', 'lstm_next_token.h5')):
        """ Train a model to predict bioactivity

        Args:
            algorithm: (Algorithm) which algorithm to use
            cv: (int) either the number of random cross-validation splits or a generator object with indices (sklearn)
            pretrained_model: (str) path to pre-trained model
            n_jobs: (int) use n cpu cores (only for classical ML methods) -1 = all cores
            config_file: (str) path to config file. If None: use default config and perform hyperparameter optimization.
            If you give a path to a non-existing config file, it will optimize hyperparameters and save them as that
            filename.
        """

        self.algorithm = algorithm

        if self.algorithm in [Algorithms.RF]:
            self.model = ml.randomforest(self.x_train, self.y_train, cv=cv, n_jobs=n_jobs, config_file=config_file,
                                         working_dir=self.data.working_dir)

        elif self.algorithm in [Algorithms.SVM, Algorithms.SVR]:
            self.model = ml.supportvector(self.x_train, self.y_train, cv=cv, n_jobs=n_jobs, config_file=config_file,
                                          working_dir=self.data.working_dir)

        elif self.algorithm in [Algorithms.KNN]:
            self.model = ml.knearestneigbour(self.x_train, self.y_train, cv=cv, n_jobs=n_jobs, config_file=config_file,
                                             working_dir=self.data.working_dir)

        elif self.algorithm in [Algorithms.GBR, Algorithms.GBM]:
            self.model = ml.gradientboosting(self.x_train, self.y_train, cv=cv, n_jobs=n_jobs, config_file=config_file,
                                             working_dir=self.data.working_dir)

        elif self.algorithm in [Algorithms.MLP]:
            self.model = mlp(self.x_train, self.y_train, self.x_val, self.y_val, config_file=config_file,
                             working_dir=self.data.working_dir)

        elif self.algorithm in [Algorithms.LSTM]:
            save_path = define_default_log_dir()
            self.model = lstm(self.x_train, self.y_train, self.x_val, self.y_val, pretrained_model=pretrained_model,
                              save_path=save_path)

        elif self.algorithm in [Algorithms.CNN]:
            self.model = smiles_cnn(self.x_train, self.y_train, self.x_val, self.y_val, save_path='Results')

        elif self.algorithm in [Algorithms.GCN, Algorithms.MPNN, Algorithms.AFP, Algorithms.GAT,
                                Algorithms.GIN_INFOMAX, Algorithms.GIN_MASKING, Algorithms.GIN_EDGEPRED,
                                Algorithms.GIN_CONTEXTPRED]:
            logs_path = define_default_log_dir()
            self.model = train_model_with_hyperparameters_optimization(self.data, config_file,
                                                                       self.algorithm, self.descriptor,
                                                                       logs_path=logs_path,
                                                                       result_path='',
                                                                       working_dir=self.data.working_dir)
        else:
            raise ValueError('Unexpected model: {}'.format(self.algorithm.value))

    def predict(self, x):
        """ Predict bioactivity on x (np.array) for all methods, SMILES (str) for graph-based methods """
        if self.algorithm in [Algorithms.RF, Algorithms.SVM, Algorithms.SVR, Algorithms.KNN, Algorithms.GBR,
                              Algorithms.GBM]:
            self.predictions = self.model.predict(x)

        if self.algorithm == Algorithms.LSTM:
            self.predictions = lstm_regression_predict(self.model, x)

        if self.algorithm in [Algorithms.CNN, Algorithms.MLP]:
            self.predictions = list(self.model.predict(x).flatten())

        if self.algorithm in [Algorithms.GCN, Algorithms.MPNN, Algorithms.AFP, Algorithms.GAT, Algorithms.GIN_INFOMAX,
                              Algorithms.GIN_MASKING, Algorithms.GIN_EDGEPRED, Algorithms.GIN_CONTEXTPRED]:
            if type(x) is not list or type(x[0]) is not str:
                NotImplementedError(f"Predictions for input type {type(x)} is not implemented for a "
                                    f"{self.algorithm.value} model. Provide a list of SMILES strings instead.")
            from MoleculeACE.GNN.models.train_test_val import predict
            self.predictions = predict(self.model, x, self.descriptor)

        return self.predictions

    def test_predict(self):
        """ Predict the bioactivity of the included test data (needs self.x_test) """
        x_test = self.data.x_test
        if self.algorithm in [Algorithms.RF, Algorithms.SVM, Algorithms.SVR, Algorithms.KNN, Algorithms.GBR,
                              Algorithms.GBM]:
            self.predictions = self.model.predict(x_test)

        if self.algorithm == Algorithms.LSTM:
            self.predictions = lstm_regression_predict(self.model, x_test)

        if self.algorithm in [Algorithms.CNN, Algorithms.MLP]:
            self.predictions = list(self.model.predict(x_test).flatten())

        if self.algorithm in [Algorithms.GCN, Algorithms.MPNN, Algorithms.AFP, Algorithms.GAT, Algorithms.GIN_INFOMAX,
                              Algorithms.GIN_MASKING, Algorithms.GIN_EDGEPRED, Algorithms.GIN_CONTEXTPRED]:
            from MoleculeACE.GNN.models.train_test_val import predict
            x_test_smiles = self.data.smiles_test
            self.predictions = predict(self.model, x_test_smiles, self.descriptor)

        return self.predictions

    def load_model(self, filename: str):
        """ Load a saved model. Tensorflow works best with .h5 files, the rest uses pickle (.pkl)"""
        if self.algorithm in [Algorithms.RF, Algorithms.SVM, Algorithms.GBM, Algorithms.GBR, Algorithms.KNN]:
            with open(filename, 'rb') as handle:
                self.model = pickle.load(handle)
        if self.algorithm in [Algorithms.LSTM]:  # Make sure the file extension in .h5 for the fastest save
            self.model = keras_load_model(filename)
        if self.algorithm in [Algorithms.CNN, Algorithms.MLP]:  # Make sure the file extension in .h5 for speed
            self.model = keras_load_model(filename)
        if self.algorithm in [Algorithms.GCN, Algorithms.MPNN, Algorithms.AFP, Algorithms.GAT, Algorithms.GIN_INFOMAX,
                              Algorithms.GIN_CONTEXTPRED, Algorithms.GIN_EDGEPRED, Algorithms.GIN_MASKING]:
            self.model = torch.load(filename)

    def save_model(self, filename: str):
        """ Save a trained model. Tensorflow works best with .h5 files, the rest uses pickle (.pkl)"""
        if self.algorithm in [Algorithms.RF, Algorithms.SVM, Algorithms.SVR, Algorithms.KNN, Algorithms.GBR,
                              Algorithms.GBM]:
            with open(filename, 'wb') as handle:
                # Pickle best_estimator_ if we used cross fold validation, else pickle the model
                try:
                    pickle.dump(self.model.best_estimator_, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.algorithm in [Algorithms.GCN, Algorithms.MPNN, Algorithms.AFP, Algorithms.GAT, Algorithms.GIN_INFOMAX,
                              Algorithms.GIN_CONTEXTPRED, Algorithms.GIN_EDGEPRED, Algorithms.GIN_MASKING]:
            torch.save(self.model, filename)
        if self.algorithm == Algorithms.LSTM:  # Make sure the file extension in .h5 for the fastest save
            self.model.save(filename)
        if self.algorithm in [Algorithms.CNN,
                              Algorithms.MLP]:  # Make sure the file extension in .h5 for the fastest save
            self.model.save(filename)

    def __repr__(self):
        return f"A {'' if self.model is None else 'Trained '}{self.algorithm.value} model"
