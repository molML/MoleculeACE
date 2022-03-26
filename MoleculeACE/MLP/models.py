"""
Code for running and optimizing a 'vanilla' deep neural network
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.random import set_seed
import numpy as np
import os
from MoleculeACE.benchmark import utils
from MoleculeACE.benchmark.utils import get_config
from MoleculeACE.benchmark.utils.const import RANDOM_SEED, CONFIG_PATH_MLP, CONFIG_PATH_GENERAL, WORKING_DIR
from MoleculeACE.benchmark.data_processing.preprocessing.data_prep import split_binary_fingerprint_array
from sklearn.model_selection import train_test_split

general_settings = get_config(CONFIG_PATH_GENERAL)
mlp_settings = get_config(CONFIG_PATH_MLP)


class NeuralNet:
    """ Define and build a simple feed forward neural net"""
    def __init__(self, descriptor_size=1024, hidden_size=256, dropout=0.5, lr=0.0001, regularization=0.01,
                 batch_size=32):
        self.model = None
        self.descriptor_size = descriptor_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.regular_l2 = regularization
        self.lr = lr
        self.batch_size = batch_size

        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_size, activation="relu", kernel_regularizer=l2(self.regular_l2),
                             input_dim=self.descriptor_size))
        self.model.add(Dropout(self.dropout))

        self.model.add(Dense(self.hidden_size, activation="relu", kernel_regularizer=l2(self.regular_l2)))
        self.model.add(Dropout(self.dropout))

        self.model.add(Dense(self.hidden_size, activation="relu", kernel_regularizer=l2(self.regular_l2)))
        self.model.add(Dropout(self.dropout))

        self.model.add(Dense(1))

        optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)


def mlp(x_train, y_train, x_val=None, y_val=None, save_path='Results/', config_file=None, verbose=1,
        working_dir=WORKING_DIR):

    if config_file is None or not os.path.exists(config_file):
        config = get_config(CONFIG_PATH_MLP)
        model = grid_search(x_train, y_train, x_val, y_val, args=config, config_filename=config_file,
                            folds=general_settings['num_cv_folds'], save_path=save_path, verbose=0,
                            working_dir=working_dir)
    else:
        config = utils.get_config(config_file)
        model = train_mlp(x_train, y_train, x_val=x_val, y_val=y_val, save_path=save_path, args=config, verbose=verbose)

    return model


def train_mlp(x_train, y_train, x_val=None, y_val=None, save_path='Results/', args=None, verbose=1):
    """ Train a neural net on molecular descriptors

    Args:
        args: (dict) hyperparameters: dropout, lr, batch_size, epochs, patience_stopping, monitor, patience_lr,
              lr_factor, min_lr, n_workers, val_split, regularization, hidden_size
        x_train: (array) array of training data
        y_train: (lst) List of bioactivity values
        y_val: (array) array of validation data (default=None)
        x_val: (lst) List of validation bioactivity values (default=None)
        save_path: (str) path where best models are stored
        verbose: (int) 0 or 1 if you want to see every epoch getting trained

    Returns: MoleculeACE.CNN.model.NeuralNet

    """
    # General settings
    epochs = mlp_settings['epochs']
    n_workers = general_settings['num_workers']
    patience_stopping = mlp_settings['early_stopping_patience']
    val_split = mlp_settings['val_split']

    dropout = args['dropout']
    lr = args['lr']
    batch_size = args['batch_size']
    monitor = args['monitor']
    patience_lr = args['patience_lr']
    lr_factor = args['lr_factor']
    min_lr = args['min_lr']
    regularization = args['regularization']
    hidden_size = args['hidden_size']

    # Set random seed
    set_seed(RANDOM_SEED)

    # convert the list of train labels to an array
    y_train = np.array(y_train)
    if y_val is not None:
        y_val = np.array(y_val)

    # find smiles array dimensions
    descriptor_size = x_train.shape[1]

    # Callbacks
    early_stopping = EarlyStopping(monitor=monitor, mode='min', verbose=1, patience=patience_stopping)
    checkpointer = utils.create_model_checkpoint(f"{save_path}/mlp_")
    lr_reduction = ReduceLROnPlateau(monitor=monitor,
                                     patience=patience_lr,
                                     factor=lr_factor,
                                     min_lr=min_lr)

    # Initiate and compile the model
    model = NeuralNet(descriptor_size=descriptor_size,
                      regularization=regularization,
                      hidden_size=hidden_size, dropout=dropout,
                      lr=lr, batch_size=batch_size).model

    # Train the model
    if x_val is None:
        if verbose:
            print(f"Creating a validation split from the train data ({val_split}) by fingerprint clusters")
        try:
            # Split by binary fingerprint
            train_idx, val_idx = split_binary_fingerprint_array(x_train, test_split=val_split, clustering_cutoff=0.4)

            x_val = x_train[val_idx]
            x_train = x_train[train_idx]
            y_val = y_train[val_idx]
            y_train = y_train[train_idx]
        except:
        # Otherwise split randomly
            print('Could not split by fingerprint (they were probably not binary) - random splitting instead')
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split,
                                                              random_state=RANDOM_SEED)

        # history = model.fit(x=x_train, y=y_train, validation_split=val_split, use_multiprocessing=True,
        #                           epochs=epochs, callbacks=[checkpointer, lr_reduction, early_stopping],
        #                           workers=n_workers, verbose=verbose)
    # else:
    history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), use_multiprocessing=True,
                              epochs=epochs, callbacks=[checkpointer, lr_reduction, early_stopping],
                              workers=n_workers, verbose=verbose)

    # plot the train/val loss
    if verbose:
        utils.plot_history(history, 'MSE')

    return model


def get_grid(args):
    """ Make a dict that contains every combination of hyperparameters to explore"""
    import itertools

    # Make a list of every value in the dict if its not
    for k, v in args.items():
        if type(v) is not list:
            args[k] = [v]
    # Get every pairwise combination
    hyperparam, values = zip(*args.items())
    grid = [dict(zip(hyperparam, v)) for v in itertools.product(*values)]

    return grid


def grid_search(x_train, y_train, x_val=None, y_val=None, args=None, config_filename: str = None, folds: int = 5,
                verbose: int = 0, default_config: str = CONFIG_PATH_MLP, working_dir: str = WORKING_DIR,
                save_path: str = os.path.join(WORKING_DIR, 'logs')):
    """ Optimize MLP through grid searching

    Args:
        working_dir: (str) path to working directory
        save_path: (str) where to store intermediate stuff
        config_filename: (str) path to config file. Best hyperparameters will be saved under this name
        x_train: array of train data
        y_train: array of train data labels, default = None
        x_val: array of validation data
        y_val: array of validation labels, default = None
        args: (dict) hyperparameters of the MLP (see config file)
        default_config: (str) path to default config file
        verbose: (bool) printout all training steps

    Returns: trained model with best hyperparameters

    """

    from sklearn.cluster import KMeans
    from sklearn.model_selection import StratifiedKFold

    # If no args are given, get the default settings to optimize
    if args is None:
        args = utils.get_config(default_config)

    if type(y_train) is list:
        y_train = np.array(y_train)

    # Create the grid-search grid
    grid = get_grid(args)

    print(f"--- Optimizing {5*len(grid)} models with {folds}-fold cross-validation")

    # Get 5 shuffled train/test folds based on kmeans clustering of the training molecules
    kmeans = KMeans(n_clusters=10, random_state=RANDOM_SEED).fit(x_train)
    skf = StratifiedKFold(n_splits=folds, random_state=RANDOM_SEED, shuffle=True)
    folds = [{'train': tr, 'test': tst} for tr, tst in skf.split(x_train, kmeans.labels_)]

    # Train n folds for every set of hyperparameters
    results = []
    for hyperparameters in grid:

        results_folds = []
        for fold in folds:
            # Get data from this split.
            fold_x_train = x_train[fold['train']]
            fold_y_train = y_train[fold['train']]
            fold_x_test = x_train[fold['test']]
            fold_y_test = y_train[fold['test']]

            # Train model
            model = train_mlp(fold_x_train, fold_y_train, args=hyperparameters, verbose=verbose, save_path=save_path)

            # Predictions
            predictions = list(model.predict(fold_x_test).flatten())
            res = np.square(np.subtract(fold_y_test, predictions)).mean()
            results_folds.append(res)
        # Append the mean test results from the n fold cross-validation + corresponding hyperparameters to a list
        results.append((np.mean(results_folds), hyperparameters))

    # Get the best hyperparameters
    results.sort(key=lambda x: x[0])
    best_hyperparamters = results[0][1]
    # Write them as a yml
    if config_filename is None:
        config_filename = os.path.join(working_dir, 'configures', 'MLP.yml')
    utils.write_config(config_filename, best_hyperparamters)

    # Train a model on all training data with the best hyperparameters
    model = train_mlp(x_train, y_train, x_val, y_val, args=best_hyperparamters, verbose=verbose, save_path=save_path)

    return model
