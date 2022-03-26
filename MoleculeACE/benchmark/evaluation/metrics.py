"""
Some evaluation metrics
Derek van Tilborg, Eindhoven University of Technology, March 2022
"""

import numpy as np


def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    return np.sqrt(np.mean(np.square(np.array(true) - np.array(pred))))


def calc_q2f3(true, pred, y_train):
    """ Calculates the Q2 F3 score (best according to Todeschini et al. 2016)

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)
        y_train: (1d array-like shape) true train values (float)

    Returns: Q2F3 score
    """

    mean_y_train = np.mean(y_train)
    n_out = len(pred)
    n_tr = len(y_train)
    press = np.sum([(true[i] - pred[i]) ** 2 for i in range(n_out)])
    tss = np.sum([(y_train[i] - mean_y_train) ** 2 for i in range(n_tr)])

    q2f3 = 1 - ((press / n_out) / (tss / n_tr))

    return q2f3
