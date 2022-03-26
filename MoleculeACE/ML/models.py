"""
Code for running and optimizing some classical machine learning algorithsm
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from MoleculeACE.benchmark import utils
import os

from MoleculeACE.benchmark.utils.const import RANDOM_SEED, CONFIG_PATH, WORKING_DIR
from MoleculeACE.benchmark.utils.const import CONFIG_PATH_RF, CONFIG_PATH_GBM, CONFIG_PATH_SVM, CONFIG_PATH_KNN


def gradientboosting(train_data, train_labels, cv=5, n_jobs=-1, config_file=None, random_state=RANDOM_SEED,
                     default_config=CONFIG_PATH_GBM, working_dir=WORKING_DIR):
    """ Optimize and train a Gradient Boosting Regressor

    Args:
        train_data: (np.array) The training data
        train_labels: (lst)  list of labels (float)
        cv:
        n_jobs: (int) number of cpu cores to use, -1 uses all
        n_estimators: (int) The number of boosting stages to perform
        max_features: (str/int) number of features to consider when looking for the best split. sqrt(n_features)
        max_depth: (int) Maximum depth of the individual regression estimators
        learning_rate: (float) Learning rate shrinks the contribution of each tree
        min_samples_split: (int) The minimum number of samples required to split an internal node
        min_samples_leaf: (int) minimum number of samples required to be at a leaf node
        max_leaf_nodes: (int) Grow trees with max_leaf_nodes in best-first fashion

    Returns: sklearn model

    """
    if config_file is None or not os.path.exists(config_file):
        config = utils.get_config(default_config)
    else:
        config = utils.get_config(config_file)

    # If there are config parameters given as a list, result to a grid search
    if len([len(v) for k, v in config.items() if type(v) is list]) > 0:
        # Put every param into a list if it isn't yet
        config = {k: utils.to_list(v) for k, v in config.items()}
        # gridsearch for the best svm parameters, use all cpu cores
        model = GridSearchCV(GradientBoostingRegressor(random_state=random_state), param_grid=config,
                             n_jobs=n_jobs, cv=cv, verbose=1)
    else:
        model = GradientBoostingRegressor(random_state=random_state, **config)

    model.fit(train_data, train_labels)

    if config_file is not None:
        if not os.path.exists(config_file) and config_file != default_config:
            utils.write_config(config_file, model.best_params_)
    if config_file is None:
        utils.write_config(os.path.join(working_dir, 'configures', 'GBM.yml'), model.best_params_)

    return model


def randomforest(train_data, train_labels, cv=5, n_jobs=-1, config_file=None, random_state=RANDOM_SEED,
                 default_config=CONFIG_PATH_RF, working_dir=WORKING_DIR):
    """ Train a random forest model. Parameters given as a list will be used for grid search

    Args:
        train_data: (np.array) The training data
        train_labels: (lst)  list of labels (float)
        cv: (int) number of cross-validation splits or  MoleculeACE.data_prep.Data.get_cv_folds()
        n_jobs: (int) number of cpu cores to use, -1 uses all
        max_features: (str) max_features=sqrt(n_features)
        n_estimators: (int) The number of trees in the forest.
        max_features_scalars: (float) multiply n_features by a scalar, overrides max_features
        max_depth: (int) the maximum depth of the tree
        min_samples_split: (int) minimum number of samples required to split an internal node
        min_samples_leaf: (int) minimum number of samples required to be at a leaf node
        max_leaf_nodes: (int) max number of leaves per node, None = unlimited leaf nodes

    Returns: sklearn model

    """
    if config_file is None or not os.path.exists(config_file):
        config = utils.get_config(default_config)
    else:
        config = utils.get_config(config_file)

    # create grid search parameters
    if 'max_features_scalars' in config and config['max_features_scalars'] is not None:
        config['max_features'] = [round(train_data.shape[1] * i) for i in utils.to_list(config['max_features_scalars'])]
        # Deconvert it from a list if it only has one value
        if len(config['max_features']) == 1:
            config['max_features'] = config['max_features'][0]
        # remove it from the config, its not readible by sklearn
        config.pop("max_features_scalars")
    else:
        config['max_features'] = 'sqrt'

    # If there are config parameters given as a list, result to a grid search
    if len([len(v) for k, v in config.items() if type(v) is list]) > 0:
        # Put every param into a list if it isn't yet
        config = {k: utils.to_list(v) for k, v in config.items()}
        # gridsearch for the best svm parameters, use all cpu cores
        model = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid=config,
                             n_jobs=n_jobs, cv=cv, verbose=1)
    else:
        model = RandomForestRegressor(random_state=random_state, **config)

    model.fit(train_data, train_labels)

    if config_file is not None:
        if not os.path.exists(config_file) and config_file != default_config:
            utils.write_config(config_file, model.best_params_)
    if config_file is None:
        utils.write_config(os.path.join(working_dir, 'configures', 'RF.yml'), model.best_params_)

    return model


def supportvector(train_data, train_labels, cv=5, n_jobs=-1, config_file=None,
                  default_config=CONFIG_PATH_SVM, working_dir=WORKING_DIR):
    """ Train a support vector model. Parameters given as a list will be used for grid search

    Args:
        train_data: (np.array) The training data
        train_labels: (lst)  list of labels (float)
        cv: (int) number of cross-validation splits or  MoleculeACE.data_prep.Data.get_cv_folds()
        n_jobs: (int) number of cpu cores to use, -1 uses all
        kernel: (str) kernel type to be used. ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
        c: (int) Regularization parameter. Must be strictly positive. L2 penalty.
        gamma: (float) Kernel coefficient
        epsilon: (float) Epsilon in the epsilon-SVR model

    Returns: sklearn model

    """
    if config_file is None or not os.path.exists(config_file):
        config = utils.get_config(default_config)
    else:
        config = utils.get_config(config_file)

    # If there are config parameters given as a list, result to a grid search
    if len([len(v) for k, v in config.items() if type(v) is list]) > 0:
        # Put every param into a list if it isn't yet
        config = {k: utils.to_list(v) for k, v in config.items()}
        # gridsearch for the best svm parameters, use all cpu cores
        model = GridSearchCV(SVR(), param_grid=config, n_jobs=n_jobs, cv=cv, verbose=1)
    else:
        model = SVR(**config)

    model.fit(train_data, train_labels)

    if config_file is not None:
        if not os.path.exists(config_file) and config_file != default_config:
            utils.write_config(config_file, model.best_params_)
    if config_file is None:
        utils.write_config(os.path.join(working_dir, 'configures', 'SVM.yml'), model.best_params_)

    return model


def knearestneigbour(train_data, train_labels, cv=5, n_jobs=-1, config_file=None,
                     default_config=CONFIG_PATH_KNN, working_dir=WORKING_DIR):
    """ Train a k-nearest-neigbour model. Parameters given as a list will be used for grid search

    Args:
        train_data: (np.array) The training data
        train_labels: (lst)  list of labels (float)
        cv: (int) number of cross-validation splits or  MoleculeACE.data_prep.Data.get_cv_folds()
        n_jobs: (int) number of cpu cores to use, -1 uses all
        n_neighbors: (int) Number of neighbors to use by default for kneighbors queries.
        weights: (str) Weight function used in prediction
        metric: (str) The distance metric to use

    Returns: sklearn model

    """
    if config_file is None or not os.path.exists(config_file):
        config = utils.get_config(default_config)
    else:
        config = utils.get_config(config_file)

    # If there are config parameters given as a list, result to a grid search
    if len([len(v) for k, v in config.items() if type(v) is list]) > 0:
        # Put every param into a list if it isn't yet
        config = {k: utils.to_list(v) for k, v in config.items()}
        # gridsearch for the best svm parameters, use all cpu cores
        model = GridSearchCV(KNeighborsRegressor(), param_grid=config, n_jobs=n_jobs, cv=cv, verbose=1)
    else:
        model = KNeighborsRegressor(**config)

    model.fit(train_data, train_labels)

    if config_file is not None:
        if not os.path.exists(config_file) and config_file != default_config:
            utils.write_config(config_file, model.best_params_)
    if config_file is None:
        utils.write_config(os.path.join(working_dir, 'configures', 'KNN.yml'), model.best_params_)

    return model
