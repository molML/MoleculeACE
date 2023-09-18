"""
Author: Derek van Tilborg -- TU/e -- June 2022

Inspired by:

Moret, M., Grisoni, F., Katzberger, P. & Schneider, G.
Perplexity-based molecule ranking and bias estimation of chemical language models.
ChemRxiv (2021) doi:10.26434/chemrxiv-2021-zv6f1-v2.

    - LSTM:                         LSTM model class
        - train()
        - test()
        - predict()
    - LSTMNextToken:                LSTM autoregressive model for pre-training
        - train()
        - test()
        - predict()
    - DataGeneratorNextToken:       Generator class for the next-token model
    - DataGeneratorRegression:      Generator class for the regression LSTM

"""

from MoleculeACE.models.utils import get_config
from MoleculeACE.benchmark.const import CONFIG_PATH_SMILES, CONFIG_PATH_LSTM, WORKING_DIR
from tensorflow.keras.layers import Dense, TimeDistributed, BatchNormalization
from tensorflow.keras.layers import LSTM as KerasLSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
import tensorflow.keras.optimizers
from MoleculeACE.benchmark.featurization import OneHotEncodeSMILES
import numpy as np
import os
import warnings


lstm_settings = get_config(CONFIG_PATH_LSTM)
smiles_encoding = get_config(CONFIG_PATH_SMILES)


class LSTM:
    def __init__(self, pretrained_model: str = os.path.join(WORKING_DIR, "pretrained_models", "pretrained_lstm.h5"),
                 nchar_in: int = 41, hidden_0: int = 1024, hidden_1: int = 256, best_model_save_path: str = os.path.join('.', 'best_model.h5'),
                 dropout: float = 0.4, lr: float = 0.0005, epochs: int = 100, batch_size: int = 32, *args, **kwargs):

        self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate=lr)
        self.epochs = epochs
        self.name = 'LSTM'
        self.save_path = best_model_save_path
        self.history = None
        self.batch_size = batch_size

        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                next_token_model = load_model(pretrained_model)

                self.model = Sequential()
                for layer in next_token_model.layers[:-1]:  # go through until last layer
                    self.model.add(layer)
                # Freeze the first LSTM layer
                self.model.layers[1].trainable = False
                self.model.layers[2].trainable = True
                # Add a dense regression layer at the end
                self.model.add(Dense(1, activation='linear'))

                self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')

                # Copy the weights from the pretrained model and paste them on all layers except for the last one.
                for idx, layer in enumerate(next_token_model.layers[:-1]):  # go through until last layer
                    self.model.layers[idx].set_weights(layer.get_weights())
            else:
                warnings.warn(f"Could not find the pre_trained LSTM model: {pretrained_model}")
        else:
            self.model = Sequential()
            self.model.add(BatchNormalization(input_shape=(None, nchar_in)))

            self.model.add(
                KerasLSTM(hidden_0, unit_forget_bias=True, dropout=dropout, trainable=True, return_sequences=True))
            self.model.add(
                KerasLSTM(hidden_1, unit_forget_bias=True, dropout=dropout, trainable=True, return_sequences=True))

            self.model.add(BatchNormalization())
            self.model.add(Dense(1, activation='linear'))

            optimizer = tensorflow.keras.optimizers.Adam(learning_rate=lr)
            self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train(self, x_train, y_train, x_val=None, y_val=None,
              early_stopping_patience: int = None, epochs: int = None, print_every_n: int = 100):

        if epochs is None:
            epochs = self.epochs

        tr_generator = DataGeneratorRegression(x_train, y_train, batch_size=self.batch_size, shuffle=True)

        if x_val is not None:
            early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_patience)
            checkpointer = ModelCheckpoint(filepath=self.save_path, verbose=0, save_best_only=True)
            val_generator = DataGeneratorRegression(x_val, y_val, batch_size=self.batch_size, shuffle=False)

            # Train model
            self.history = self.model.fit(tr_generator, validation_data=val_generator, use_multiprocessing=True,
                                          epochs=epochs, callbacks=[checkpointer, early_stopping],
                                          workers=1, verbose=1)
        else:
            # Train model
            self.history = self.model.fit(tr_generator, use_multiprocessing=True, epochs=epochs, workers=1, verbose=1)

    def test(self, x_test, y_test, batch_size: int = 32):
        y_hat = self.predict(x_test, batch_size)

        return y_hat, y_test

    def predict(self, x, batch_size: int = 32):

        # Add some empty samples to the test data to make the test data divisible by the batch size
        n_expand = batch_size - x.shape[0] % batch_size
        empty_arr = np.empty([n_expand, x.shape[1], x.shape[2]])
        expanded_data = np.append(x, empty_arr, axis=0)

        # Construct a generator with dummy y labels
        test_generator = DataGeneratorRegression(expanded_data, [0] * len(expanded_data), batch_size, shuffle=False)

        # Go through the generator and take the last item for each prediction for prediction p in each batch b
        # This is also what happens under the bonnet of the tensorflow LSTM
        predictions = sum([[p[-1][0] for p in self.model.predict(b[0])] for b in test_generator], [])

        return predictions[:len(x)]  # Discard the empty samples, as they are nonsense predictions

    def __repr__(self):
        return 'LSTM'


class LSTMNextToken:
    def __init__(self, nchar_in: int = 41, hidden_0: int = 1024, hidden_1: int = 256,
                 dropout: float = 0.4, lr: float = 0.0005, epochs: int = 100, *args, **kwargs):

        self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate=lr)
        self.epochs = epochs
        self.name = 'LSTMNextToken'
        self.save_path = os.path.join('.', 'best_model.h5')
        self.history = None

        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=(None, nchar_in)))

        self.model.add(
            KerasLSTM(hidden_0, unit_forget_bias=True, dropout=dropout, trainable=True, return_sequences=True))
        self.model.add(
            KerasLSTM(hidden_1, unit_forget_bias=True, dropout=dropout, trainable=True, return_sequences=True))

        self.model.add(BatchNormalization())
        self.model.add(TimeDistributed(Dense(nchar_in, activation='softmax')))

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)

    def train(self, x_train, x_val=None, early_stopping_patience: int = None, epochs: int = None, batch_size: int = 32):

        if epochs is None:
            epochs = self.epochs
        if early_stopping_patience is None:
            early_stopping_patience = epochs

        tr_generator = DataGeneratorNextToken(x_train, batch_size=batch_size, shuffle=True)
        val_generator = DataGeneratorNextToken(x_val, batch_size=batch_size, shuffle=False)

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_patience)
        checkpointer = ModelCheckpoint(filepath=self.save_path, verbose=0, save_best_only=True)

        self.history = self.model.fit(tr_generator,
                                      validation_data=val_generator,
                                      use_multiprocessing=True,
                                      epochs=epochs,
                                      callbacks=[checkpointer, early_stopping],
                                      workers=1, verbose=1)

    def test(self, x_test, batch_size: int = 32):

        test_generator = DataGeneratorNextToken(x_test, batch_size=batch_size, shuffle=False)

        # predict in batches, otherwise you'll get giant tensors
        predicted_chars = []
        true_chars = []
        for batch in test_generator:
            predicted_chars.append(self.model.predict(batch[0]).argmax(2).flatten())
            true_chars.append(np.array(batch[1]).argmax(2).flatten())

        # flatten list of predictions
        predicted_chars = np.array(predicted_chars).flatten()
        true_chars = np.array(true_chars).flatten()

        return predicted_chars, true_chars

    def predict(self, x, batch_size: int = 32):
        return self.test(x, batch_size)[0]

    def __repr__(self):
        return 'Next-token LSTM model'


class DataGeneratorNextToken(Sequence):
    """Generates one-hot encoded smiles + next token data for Keras"""

    def __init__(self, smiles, batch_size: int = 32, max_len_model: int = smiles_encoding['max_smiles_len'] + 2,
                 n_chars: int = smiles_encoding['vocab_size'], indices_token: dict = smiles_encoding['indices_token'],
                 token_indices: dict = smiles_encoding['token_indices'], shuffle: bool = True):
        """Initialization"""
        self.max_len_model = max_len_model
        self.batch_size = batch_size
        self.smiles = smiles
        self.shuffle = shuffle
        self.n_chars = n_chars
        self.onehotfeaturizer = OneHotEncodeSMILES()

        self.on_epoch_end()

        self.indices_token = indices_token
        self.token_indices = token_indices

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.smiles) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.smiles))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates batch of data containing batch_size samples"""

        switch = 1
        y = np.empty((self.batch_size, self.max_len_model - switch, self.n_chars), dtype=int)
        x = np.empty((self.batch_size, self.max_len_model - switch, self.n_chars), dtype=int)

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            smi = self.onehotfeaturizer(self.smiles[ID])
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
