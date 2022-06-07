"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

A collection of classical machine learning algorithms

    -MLmodel:       parent class used by all machine learning algorithms
    -RF:            Random Forest Regressor
    -GBM:           Gradient Boosting Regressor
    -SVM:           Support Vector Regressor
    -KNN:           K-Nearest neighbour Regressor

"""

import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from numpy.typing import ArrayLike
from typing import List, Union, Dict


class MLmodel:
    def __init__(self):
        self.model = None
        self.epoch = None

    def predict(self, x: ArrayLike, *args, **kwargs):
        return self.model.predict(x)

    def train(self, x_train: ArrayLike, y_train: Union[List[float], ArrayLike], *args, **kwargs):
        if type(y_train) is list:
            y_train = np.array(y_train)
        self.model.fit(x_train, y_train)

    def test(self, x_test: ArrayLike, y_test: Union[List[float], ArrayLike], *args, **kwargs):
        if type(y_test) is list:
            y_test = np.array(y_test)
        y_hat = self.model.predict(x_test)

        return y_hat, y_test

    def __call__(self, x: ArrayLike):
        return self.model.predict(x)


class RF(MLmodel):
    def __init__(self, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__()
        self.model = RandomForestRegressor(**hyperparameters)
        self.name = 'RF'

    def __repr__(self):
        return f"Random Forest Regressor: {self.model.get_params()}"


class GBM(MLmodel):
    def __init__(self, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__()
        self.model = GradientBoostingRegressor(**hyperparameters)
        self.name = 'GBM'

    def __repr__(self):
        return f"Gradient Boosting Regressor: {self.model.get_params()}"


class SVM(MLmodel):
    def __init__(self, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__()
        self.model = SVR(**hyperparameters)
        self.name = 'SVR'

    def __repr__(self):
        return f"Support Vector Regressor: {self.model.get_params()}"


class KNN(MLmodel):
    def __init__(self, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__()
        self.model = KNeighborsRegressor(**hyperparameters)
        self.name = 'KNN'

    def __repr__(self):
        return f"K-Nearest Neighbour regressor: {self.model.get_params()}"
