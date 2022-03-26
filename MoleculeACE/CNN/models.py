"""
Code for running a 1D-Convolutional neural network on SMILES strings
Derek van Tilborg, Eindhoven University of Technology, March 2022

inspired by:

Kimber, T. B., Gagnebin, M. & Volkamer, A. Maxsmi: Maximizing molecular property prediction performance with confidence
estimation using SMILES augmentation and deep learning. Artificial Intelligence in the Life Sciences 1, 100014 (2021).

SMILES encoding settings are adapted from:

Moret, M., Grisoni, F., Katzberger, P. & Schneider, G.
Perplexity-based molecule ranking and bias estimation of chemical language models.
ChemRxiv (2021) doi:10.26434/chemrxiv-2021-zv6f1-v2.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution1D, Dropout, BatchNormalization, MaxPooling1D
import tensorflow.keras.optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.random import set_seed
import numpy as np

from MoleculeACE.benchmark import utils
from MoleculeACE.benchmark.utils.const import RANDOM_SEED
from MoleculeACE.benchmark.utils.const import CONFIG_PATH_GENERAL, CONFIG_PATH_SMILES, CONFIG_PATH_CNN
from MoleculeACE.benchmark.data_processing.preprocessing.one_hot_encoding import onehot_to_smiles
from MoleculeACE.benchmark.data_processing.preprocessing.data_prep import split_smiles

cnn_settings = utils.get_config(CONFIG_PATH_CNN)
smiles_encoding = utils.get_config(CONFIG_PATH_SMILES)
general_settings = utils.get_config(CONFIG_PATH_GENERAL)


class SeqModel:
    """ Define and build the 1D CNN """
    def __init__(self, smiles_length=202, n_chars=35, dropout=0.5, lr=0.0001, batch_size=32):
        self.model = None
        self.smiles_length = smiles_length
        self.n_chars = n_chars
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size

        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Convolution1D(32, 1, activation="relu", input_shape=(self.smiles_length, self.n_chars)))
        self.model.add(MaxPooling1D(2))

        self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(1))

        optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)


def smiles_cnn(x_train, y_train, x_val=None, y_val=None, save_path='Results/'):
    """ Train a 1D CNN on one-hot encoded smiles data

    Args:
        x_train: (array) one-hot encoded smiles with shape (n_samples, max_smiles_len, n_characters)
        y_train: (lst) List of bioactivity values
        y_val: (array) one-hot encoded validation smiles with shape (n_samples, max_smiles_len, n_char) (default=None)
        x_val: (lst) List of validation bioactivity values (default=None)
        epochs: (int) number of epochs the model will be trained
        save_path: (str) path where best models are stored

    Returns: MoleculeACE.CNN.model.SeqModel

    """

    # Get the hyperparameters from the cnn config file
    dropout = cnn_settings['dropout']
    lr = cnn_settings['lr']
    batch_size = cnn_settings['batch_size']
    patience_stopping = cnn_settings['early_stopping_patience']
    monitor = cnn_settings['monitor']
    patience_lr = cnn_settings['patience_lr']
    lr_factor = cnn_settings['lr_factor']
    min_lr = cnn_settings['min_lr']
    n_workers = general_settings['num_workers']
    val_split = cnn_settings['val_split']

    # Set random seed
    set_seed(RANDOM_SEED)

    # convert the list of train labels to an array
    y_train = np.array(y_train)
    if y_val is not None:
        y_val = np.array(y_val)

    # find smiles array dimensions
    smiles_length = x_train.shape[1]
    n_chars = x_train.shape[2]

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                   patience=patience_stopping)
    checkpointer = utils.create_model_checkpoint(f"{save_path}/cnn_")
    lr_reduction = ReduceLROnPlateau(monitor=monitor,
                                     patience=patience_lr,
                                     factor=lr_factor,
                                     min_lr=min_lr)

    # Initiate and compile the model
    model = SeqModel(smiles_length=smiles_length, n_chars=n_chars, dropout=dropout, lr=lr, batch_size=batch_size).model

    # Train the model
    if x_val is None and y_val is None:
        print(f"Creating a validation split from the train data ({val_split})")
        train_smiles = []
        for array in x_train:
            smi = onehot_to_smiles(array, smiles_encoding['indices_token'],
                                   smiles_encoding['start_char'],
                                   smiles_encoding['end_char'])
            train_smiles.append(smi)

        # Split based on Murco scaffold clustering
        train_idx, val_idx = split_smiles(train_smiles, test_split=val_split, clustering_cutoff=0.4)

        # Split the train data into train and validation
        x_val = x_train[val_idx]
        x_train = x_train[train_idx]
        y_val = y_train[val_idx]
        y_train = y_train[train_idx]

    history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), use_multiprocessing=True,
                              epochs=cnn_settings['epochs'],
                              callbacks=[checkpointer, lr_reduction, early_stopping],
                              workers=n_workers, verbose=1)

    # plot the train/val loss
    utils.plot_history(history, 'MSE')

    return model
