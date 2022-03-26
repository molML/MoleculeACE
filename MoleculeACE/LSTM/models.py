"""
Code to run and pre-train LSTM models on SMILES strings
Derek van Tilborg, Eindhoven University of Technology, March 2022

Inspired by:

Moret, M., Grisoni, F., Katzberger, P. & Schneider, G.
Perplexity-based molecule ranking and bias estimation of chemical language models.
ChemRxiv (2021) doi:10.26434/chemrxiv-2021-zv6f1-v2.
"""

import tensorflow.keras.optimizers
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import os

from MoleculeACE.benchmark.data_processing.preprocessing.one_hot_encoding import onehot_to_smiles
from MoleculeACE.benchmark.data_processing.preprocessing.data_prep import split_smiles
from MoleculeACE.benchmark import utils, data_processing
from MoleculeACE.LSTM import dataloaders
from MoleculeACE.benchmark.utils.const import RANDOM_SEED
from MoleculeACE.benchmark.utils.const import CONFIG_PATH_GENERAL, CONFIG_PATH_SMILES, CONFIG_PATH_LSTM

lstm_settings = utils.get_config(CONFIG_PATH_LSTM)
smiles_encoding = utils.get_config(CONFIG_PATH_SMILES)
general_settings = utils.get_config(CONFIG_PATH_GENERAL)


class SeqModel:
    """Class to define the language model, i.e the neural net"""

    def __init__(self, n_chars, max_length, layers, dropouts, trainables, lr):
        self.n_chars = n_chars
        self.max_length = max_length

        self.layers = layers
        self.dropouts = dropouts
        self.trainables = trainables
        self.lr = lr

        self.model = None
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=(None, self.n_chars)))

        for neurons, dropout, trainable in zip(self.layers, self.dropouts, self.trainables):
            self.model.add(LSTM(neurons, unit_forget_bias=True, dropout=dropout,
                                trainable=trainable, return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(TimeDistributed(Dense(self.n_chars, activation='softmax')))

        optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def pretrain(extra_smiles_train=None, extra_smiles_test=None, save_path='Results/'):

    print(f"Pre-training model on all available train data with next-token prediction, this will take a several hours")
    if extra_smiles_train is None:
        print("To make sure the pre-trained model covers the chemical space of your data, \n"
              "it is highly recommended to add a list of SMILES strings of your training \n"
              "set to the pre-training data with the 'add_smiles parameter")

    print(f"Augmenting all SMILES strings {lstm_settings['augmentation']}x by adding non-canonical SMILES")

    train_smi, test_smi, val_smi = dataloaders.prep_smiles_pretrain(val_split=lstm_settings['val_split'],
                                                                    augmentation=lstm_settings['augmentation'],
                                                                    extra_smiles_train=extra_smiles_train,
                                                                    extra_smiles_test=extra_smiles_test)

    tr_generator = dataloaders.DataGeneratorNextToken(train_smi,
                                             lstm_settings['batch_size'],
                                             lstm_settings['max_len_model'],
                                             smiles_encoding['vocab_size'],
                                             smiles_encoding['indices_token'],
                                             smiles_encoding['token_indices'],
                                             shuffle=True)

    val_generator = dataloaders.DataGeneratorNextToken(val_smi,
                                              lstm_settings['batch_size'],
                                              lstm_settings['max_len_model'],
                                              smiles_encoding['vocab_size'],
                                              smiles_encoding['indices_token'],
                                              smiles_encoding['token_indices'],
                                              shuffle=False)

    test_generator = dataloaders.DataGeneratorNextToken(test_smi,
                                               lstm_settings['batch_size'],
                                               lstm_settings['max_len_model'],
                                               smiles_encoding['vocab_size'],
                                               smiles_encoding['indices_token'],
                                               smiles_encoding['token_indices'],
                                               shuffle=False)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                   patience=lstm_settings['early_stopping_patience'])
    checkpointer = utils.create_model_checkpoint(save_path)
    lr_reduction = ReduceLROnPlateau(monitor=lstm_settings['monitor'],
                                     patience=lstm_settings['patience_lr'],
                                     factor=lstm_settings['lr_factor'],
                                     min_lr=lstm_settings['min_lr'])

    # Initiate model
    seqmodel = SeqModel(smiles_encoding['vocab_size'],
                        lstm_settings['max_len_model'],
                        [lstm_settings['layer_1'], lstm_settings['layer_2']],
                        [lstm_settings['dropout_1'], lstm_settings['dropout_2']],
                        [lstm_settings['train_layer1'], lstm_settings['train_layer2']],
                        lstm_settings['lr'])
    # seqmodel.model.summary()

    # Train model
    history = seqmodel.model.fit(tr_generator,
                                 validation_data=val_generator,
                                 use_multiprocessing=True,
                                 epochs=lstm_settings['epochs'],
                                 callbacks=[checkpointer, lr_reduction, early_stopping],
                                 workers=general_settings['num_workers'],
                                 verbose=1)

    # Save the loss history
    seqmodel.model.save(f'Pretrained_models/lstm_next_token.h5')
    # plot the train history
    utils.plot_history(history, 'categorical crossentropy loss')

    # predict in batches, otherwise you'll get giant tensors
    predicted_chars = []
    true_chars = []
    for batch in test_generator:
        predicted_chars.append(seqmodel.model.predict(batch[0]).argmax(2).flatten())
        true_chars.append(np.array(batch[1]).argmax(2).flatten())

    # flatten list of predictions
    predicted_chars = np.array(predicted_chars).flatten()
    true_chars = np.array(true_chars).flatten()

    # Don't calculate the accuracy of predicting the padding character. This highly inflates the accuracy.
    pad_char_idx = smiles_encoding['token_indices'][smiles_encoding['pad_char']]
    acc = np.mean([int(t == p) for t, p in zip(true_chars, predicted_chars) if p != pad_char_idx])
    print(f"Accuracy = {acc:.4f}")

    # Accuracy = 0.6801
    # np.mean([int(t == p) for t, p in zip(true_chars, predicted_chars)])

    return seqmodel.model


def fresh_lstm(n_chars, layers, dropouts, trainables, lr):
    """ A clean regression lstm for training w/o pre-training"""
    model = Sequential()
    model.add(BatchNormalization(input_shape=(None, n_chars)))

    for neurons, dropout, trainable in zip(layers, dropouts, trainables):
        model.add(LSTM(neurons, unit_forget_bias=True, dropout=dropout,
                       trainable=trainable, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))

    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def convert_to_regression(model, trainables=None):
    """ Convert a next character classifier lstm model to a regression model.
    Keep and freeze the weights of the first LSTM.

    Args:
        model: A model with a LSTM layer and ends with a classification layer
        trainables: (lst) list of bools (len=2) determining if the LSTM layers are trainable

    Returns: the same model but with a regression layer and loss function.

    """
    if trainables is None:
        trainables = [False, True]

    model_ft = Sequential()
    for layer in model.layers[:-1]:  # go through until last layer
        model_ft.add(layer)
    # Freeze LSTM layers accoring to the 'trainables' param
    model_ft.layers[1].trainable = trainables[0]
    model_ft.layers[2].trainable = trainables[-1]
    # Add a dense regression layer at the end
    model_ft.add(Dense(1, activation='linear'))
    # model_ft.summary()
    model_ft.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=lstm_settings['lr']),
                     loss='mean_squared_error')

    # Take the weights from the pretrained model and apply them on all layers except for the last one.
    for idx, layer in enumerate(model.layers[:-1]):  # go through until last layer
        model_ft.layers[idx].set_weights(layer.get_weights())

    return model_ft


def lstm(x_train, y_train, x_val=None, y_val=None, pretrained_model='Pretrained_models/lstm_next_token.h5',
         save_path='Results/lstm_models', trainables=None):


    if pretrained_model is not None:
        # Load the pretrained model if the user provides one
        next_token_model = None
        try:
            print(f"Loading pre-trained model from: {pretrained_model}")
            next_token_model = load_model(pretrained_model)
        except Exception:
            FileNotFoundError("Was not able to load the pre-trained model. You can train such a model using: "
                              "MoleculeACE.LSTM.models.pretrain()")

        if trainables is None:
            trainables = [False, True]
        print("Converting the pre-trained model from a next-token predictor to a regression model")
        model = convert_to_regression(next_token_model, trainables=trainables)

    else:
        print('Skipping pre-training')
        model = fresh_lstm(smiles_encoding['vocab_size'],
                           [lstm_settings['layer_1'], lstm_settings['layer_2']],
                           [lstm_settings['dropout_1'], lstm_settings['dropout_2']],
                           [lstm_settings['train_layer1'], lstm_settings['train_layer2']],
                           lstm_settings['lr'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                   patience=lstm_settings['early_stopping_patience'])
    checkpointer = utils.create_model_checkpoint(f"{save_path}/lstm_")
    lr_reduction = ReduceLROnPlateau(monitor=lstm_settings['monitor'],
                                     patience=lstm_settings['patience_lr'],
                                     factor=lstm_settings['lr_factor'],
                                     min_lr=lstm_settings['min_lr'])

    # Prep data
    if x_val is None:
        # x_train, y_train = data.x_train, data.y_train
        print(f"take a random {lstm_settings['val_split']} split of the train data to validate while training")
        # Convert arrays to smiles (I know this is inefficient)
        train_smiles = []
        for array in x_train:
            smi = onehot_to_smiles(array, smiles_encoding['indices_token'],
                                   smiles_encoding['start_char'],
                                   smiles_encoding['end_char'])
            train_smiles.append(smi)

        # Split based on Murco scaffold clustering
        train_idx, val_idx = split_smiles(train_smiles, test_split=lstm_settings['val_split'], clustering_cutoff=0.4)

        # Split the train data into train and validation
        x_val = x_train[val_idx]
        x_train = x_train[train_idx]
        y_val = [y_train[i] for i in val_idx]
        y_train = [y_train[i] for i in train_idx]


    tr_generator = dataloaders.DataGeneratorRegression(x_train,
                                                       y_train,
                                                       lstm_settings['batch_size'],
                                                       lstm_settings['max_len_model'],
                                                       smiles_encoding['vocab_size'],
                                                       smiles_encoding['indices_token'],
                                                       smiles_encoding['token_indices'],
                                                       shuffle=True)

    val_generator = dataloaders.DataGeneratorRegression(x_val,
                                                        y_val,
                                                        lstm_settings['batch_size'],
                                                        lstm_settings['max_len_model'],
                                                        smiles_encoding['vocab_size'],
                                                        smiles_encoding['indices_token'],
                                                        smiles_encoding['token_indices'],
                                                        shuffle=False)

    # Train model
    history = model.fit(tr_generator,
                        validation_data=val_generator,
                        use_multiprocessing=True,
                        epochs=lstm_settings['epochs'],
                        callbacks=[checkpointer, lr_reduction, early_stopping],
                        workers=general_settings['num_workers'],
                        verbose=1)

    utils.plot_history(history, 'MSE')

    return model


def lstm_regression_predict(model, x):
    """ Wonky predict function for lstm predictions that takes care of issues with batch sizes"""

    # Add some empty samples to the test data to make the test data divisible by the batch size
    n_expand = lstm_settings['batch_size'] - x.shape[0] % lstm_settings['batch_size']
    empty_arr = np.empty([n_expand, x.shape[1], x.shape[2]])
    expanded_data = np.append(x, empty_arr, axis=0)

    test_generator = dataloaders.DataGeneratorRegression(expanded_data,
                                                         [0] * len(expanded_data),
                                                         lstm_settings['batch_size'],
                                                         lstm_settings['max_len_model'],
                                                         smiles_encoding['vocab_size'],
                                                         smiles_encoding['indices_token'],
                                                         smiles_encoding['token_indices'],
                                                         shuffle=False)

    # Go through the generator and take the last item for each prediction for prediction p in each batch b
    # This is also what happens under the bonnet of the tensorflow LSTM
    predictions = sum([[p[-1][0] for p in model.predict(b[0])] for b in test_generator], [])
    predictions = predictions[:len(x)]  # Discard the empty samples, as they are nonsense predictions

    return predictions
