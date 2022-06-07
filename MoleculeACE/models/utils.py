"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

Utility functions that are used by several different models.

    - GNN()                 Base GNN parent class for all graph neural network models
    - graphs_to_loader()    Turn a list of graphs and bioactivities into a dataloader

"""

from MoleculeACE.benchmark.const import CONFIG_PATH_SMILES
from MoleculeACE.benchmark.utils import get_config
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tensorflow.keras.utils import Sequence
from typing import List, Dict
import matplotlib.pyplot as plt
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset
import os
import pickle
import numpy as np


smiles_encoding = get_config(CONFIG_PATH_SMILES)


class GNN:
    """ Base GNN class that takes care of training, testing, predicting for all graph-based methods """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
        self.epochs = 100
        self.save_path = os.path.join('.', 'best_model.pkl')

        self.model = None
        self.device = None
        self.loss_fn = None
        self.optimizer = None


    def train(self, x_train: List[Data], y_train: List[float], x_val: List[Data] = None, y_val: List[float] = None,
              early_stopping_patience: int = None, epochs: int = None, print_every_n: int = 100):
        """ Train a graph neural network.

        :param x_train: (List[Data]) a list of graph objects for training
        :param y_train: (List[float]) a list of bioactivites for training
        :param x_val: (List[Data]) a list of graph objects for validation
        :param y_val: (List[float]) a list of bioactivites for validation
        :param epochs: (int) train the model for n epochs
        :param print_every_n: (int) printout training progress every n epochs
        """
        if epochs is None:
            epochs = self.epochs
        train_loader = graphs_to_loader(x_train, y_train)
        patience = None if early_stopping_patience is None else 0

        for epoch in range(epochs):

            # If we reached the end of our patience, load the best model and stop training
            if patience is not None and patience >= early_stopping_patience:

                if print_every_n < epochs:
                    print('Stopping training early')
                try:
                    with open(self.save_path, 'rb') as handle:
                        self.model = pickle.load(handle)

                    os.remove(self.save_path)
                except Warning:
                    print('Could not load best model, keeping the current weights instead')

                break

            # As long as the model is still improving, continue training
            else:
                loss = self._one_epoch(train_loader)
                self.train_losses.append(loss)

                val_loss = 0
                if x_val is not None:
                    val_pred = self.predict(x_val)
                    val_loss = self.loss_fn(squeeze_if_needed(val_pred), torch.tensor(y_val))
                self.val_losses.append(val_loss)

                self.epoch += 1

                # Pickle model if its the best
                if val_loss <= min(self.val_losses):
                    with open(self.save_path, 'wb') as handle:
                        pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    patience += 1

                if self.epoch % print_every_n == 0:
                    print(f"Epoch {self.epoch} | Train Loss {loss} | Val Loss {val_loss}")

    def _one_epoch(self, train_loader):
        """ Perform one forward pass of the train data through the model and perform backprop

        :param train_loader: Torch geometric data loader with training data
        :return: loss
        """
        # Enumerate over the data
        for idx, batch in enumerate(train_loader):

            # Move batch to gpu
            batch.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)

            # Calculating the loss and gradients
            loss = self.loss_fn(squeeze_if_needed(y_hat), squeeze_if_needed(batch.y))
            if not loss > 0:
                print(idx)

            # Calculate gradients
            loss.backward()

            # Update weights
            self.optimizer.step()

        return loss

    def test(self,  x_test: List[Data], y_test: List[float]):
        """ Perform testing

        :param x_test: (List[Data]) a list of graph objects for testing
        :param y_test: (List[float]) a list of bioactivites for testing
        :return: A tuple of two 1D-tensors (predicted, true)
        """
        data_loader = graphs_to_loader(x_test, y_test, shuffle=False)
        y_pred, y = [], []
        with torch.no_grad():
            for batch in data_loader:
                batch.to(self.device)
                pred = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                y_pred.extend(squeeze_if_needed(pred).tolist())
                y.extend(squeeze_if_needed(batch.y).tolist())

        return torch.tensor(y_pred), torch.tensor(y)

    def predict(self, x, batch_size: int = 32):
        """ Predict bioactivity on molecular graphs

        :param x_test: (List[Data]) a list of graph objects for testing
        :param batch_size: (int) batch size for the prediction loader
        :return: A 1D-tensors of predicted values
        """
        loader = DataLoader(x, batch_size=batch_size, shuffle=False)
        y_pred = []
        with torch.no_grad():
            for batch in loader:
                batch.to(self.device)
                y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                y_pred.extend(squeeze_if_needed(y_hat).tolist())

        return torch.tensor(y_pred)

    def __repr__(self):
        return 'Basic Graph Neural Network Class'



class NN:
    def __init__(self):

        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
        self.epochs = 100
        self.save_path = os.path.join('.', 'best_model.pkl')

        self.model = None
        self.device = None
        self.loss_fn = None
        self.optimizer = None

    def train(self, x_train: ArrayLike, y_train: List[float], x_val: ArrayLike = None, y_val: List[float] = None,
              early_stopping_patience: int = None, epochs: int = None, print_every_n: int = 100):

        if epochs is None:
            epochs = self.epochs
        train_loader = numpy_loader(x_train, y_train)
        patience = None if early_stopping_patience is None else 0

        for epoch in range(epochs):

            # If we reached the end of our patience, load the best model and stop training
            if patience is not None and patience >= early_stopping_patience:

                if print_every_n < epochs:
                    print('Stopping training early')
                try:
                    with open(self.save_path, 'rb') as handle:
                        self.model = pickle.load(handle)

                    os.remove(self.save_path)
                except Warning:
                    print('Could not load best model, keeping the current weights instead')

                break

            # As long as the model is still improving, continue training
            else:

                loss = self._one_epoch(train_loader)
                self.train_losses.append(loss)

                val_loss = 0
                if x_val is not None:
                    val_pred = self.predict(x_val)
                    val_loss = self.loss_fn(squeeze_if_needed(val_pred), torch.tensor(y_val))

                self.val_losses.append(val_loss)

                self.epoch += 1

                # Pickle model if its the best
                if val_loss <= min(self.val_losses):
                    with open(self.save_path, 'wb') as handle:
                        pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    patience += 1

                if self.epoch % print_every_n == 0:
                    print(f"Epoch {self.epoch} | Train Loss {loss} | Val Loss {val_loss} | patience {patience}")

    def _one_epoch(self, train_loader):
        """ Perform one forward pass of the train data through the model and perform backprop

        :param train_loader: Torch geometric data loader with training data
        :return: loss
        """
        # Enumerate over the data
        for idx, batch in enumerate(train_loader):

            # Move batch to gpu
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            y_hat = self.model(x.float())

            # Calculating the loss and gradients
            loss = self.loss_fn(squeeze_if_needed(y_hat), squeeze_if_needed(y))
            if not loss > 0:
                print(idx)

            # Calculate gradients
            loss.backward()

            # Update weights
            self.optimizer.step()

        return loss

    def test(self, x_test: ArrayLike, y_test: List[float]):
        """ Perform testing

        :param x_test: (List[Data]) a list of graph objects for testing
        :param y_test: (List[float]) a list of bioactivites for testing
        :return: A tuple of two 1D-tensors (predicted, true)
        """
        data_loader = numpy_loader(x_test, y_test)
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                pred = self.model(x.float())
                y_pred.extend([i for i in squeeze_if_needed(pred).tolist()])
                y_true.extend([i for i in squeeze_if_needed(y).tolist()])

        return torch.tensor(y_pred), torch.tensor(y)

    def predict(self, x, batch_size: int = 32):
        """ Predict bioactivity on molecular graphs

        :param x: (Array) a list of graph objects for testing
        :param batch_size: (int) batch size for the prediction loader
        :return: A 1D-tensors of predicted values
        """
        data_loader = numpy_loader(x, batch_size=batch_size)
        y_pred = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0].to(self.device)
                y_hat = self.model(x.float())
                y_pred.extend([i for i in squeeze_if_needed(y_hat).tolist()])

        return torch.tensor(y_pred)

    def __repr__(self):
        return f"Neural Network baseclass for NN taking numpy arrays"



class DataGeneratorNextToken(Sequence):
    """Generates one-hot encoded smiles + next token data for Keras"""

    def __init__(self, encoded_smiles, batch_size: int = 32, max_len_model: int = smiles_encoding['max_smiles_len'] + 2,
                 n_chars: int = smiles_encoding['vocab_size'], indices_token: dict = smiles_encoding['indices_token'],
                 token_indices: dict = smiles_encoding['token_indices'], shuffle: bool = True):
        """Initialization"""
        self.max_len_model = max_len_model
        self.batch_size = batch_size
        self.encoded_smiles = encoded_smiles
        self.shuffle = shuffle
        self.n_chars = n_chars

        self.on_epoch_end()

        self.indices_token = indices_token
        self.token_indices = token_indices

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.encoded_smiles) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.encoded_smiles))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates batch of data containing batch_size samples"""

        switch = 1
        y = np.empty((self.batch_size, self.max_len_model - switch, self.n_chars), dtype=int)
        x = np.empty((self.batch_size, self.max_len_model - switch, self.n_chars), dtype=int)

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            smi = self.encoded_smiles[ID]
            x[i] = smi[:-1]
            y[i] = smi[1:]

        return x, y


class DataGeneratorRegression(Sequence):
    """Generates one-hot encoded smiles + regression data for Keras"""

    def __init__(self, encoded_smiles, activities, batch_size: int = 32,
                 max_len_model: int = smiles_encoding['max_smiles_len'] + 2,
                 n_chars: int = smiles_encoding['vocab_size'],
                 indices_token: dict = smiles_encoding['indices_token'],
                 token_indices: dict = smiles_encoding['token_indices'], shuffle: bool = True):

        """Initialization"""
        self.max_len_model = max_len_model
        self.batch_size = batch_size
        self.encoded_smiles = encoded_smiles
        self.activities = activities
        self.shuffle = shuffle
        self.n_chars = n_chars

        self.on_epoch_end()

        self.indices_token = indices_token
        self.token_indices = token_indices

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.encoded_smiles) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        x, y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.encoded_smiles))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates batch of data containing batch_size samples"""
        switch = 1
        y = np.empty((self.batch_size, 1), dtype=float)
        x = np.empty((self.batch_size, self.max_len_model - switch, self.n_chars), dtype=int)

        # Generate data
        for i, ID in enumerate(indexes):
            x[i] = self.encoded_smiles[ID][:-1]
            y[i] = self.activities[ID]

        return x, y


class NumpyDataset(Dataset):
    def __init__(self, x: ArrayLike, y: List[float] = None):
        """ Create a dataset for the ChemBerta transformer using a pretrained tokenizer """
        super().__init__()

        if y is None:
            y = [0]*len(x)
        self.y = torch.tensor(y).unsqueeze(1)
        self.x = torch.tensor(x).float()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


def graphs_to_loader(x: List[Data], y: List[float], batch_size: int = 64, shuffle: bool = False):
    """ Turn a list of graph objects and a list of labels into a Dataloader """
    for graph, label in zip(x, y):
        graph.y = torch.tensor(label)

    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)


def plot_loss(model):
    train_losses_float = [float(loss.cpu().detach().numpy()) for loss in model.train_losses]
    val_losses_float = [float(loss) for loss in model.val_losses]
    loss_indices = [i for i, l in enumerate(train_losses_float)]

    plt.figure()
    plt.plot(loss_indices, train_losses_float, val_losses_float)
    plt.show()


def scatter(y_hat, y, min_axis_val: float = -5, max_axis_val: float = 1):

    plt.figure()
    plt.scatter(x=y_hat, y=y, alpha=0.5)
    plt.axline((0, 0), slope=1, color="black", linestyle=(1, (5, 5)))
    plt.xlim(min_axis_val, max_axis_val)
    plt.ylim(min_axis_val, max_axis_val)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.show()


def numpy_loader(x: ArrayLike, y: List[float] = None, batch_size: int = 32, shuffle: bool = False):
    dataset = NumpyDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def squeeze_if_needed(tensor):
    from torch import Tensor
    if len(tensor.shape) > 1 and tensor.shape[1] == 1 and type(tensor) is Tensor:
        tensor = tensor.squeeze()
    return tensor
