import math
import random
from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .25) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    # X["response"] = y
    # X = X.sample(frac=1)
    # shuffled_response = X["response"]
    # X.drop(["response"],
    #     inplace=True, axis=1)
    # train_samples_size = math.ceil(X.shape[0] * train_proportion)
    #
    # train_x = X[:train_samples_size]
    # train_y = shuffled_response[:train_samples_size]
    # test_x = X[train_samples_size:]
    # test_y =shuffled_response[train_samples_size:]

    train_x = X.sample(frac=train_proportion)
    train_y = y[train_x.index]
    test_x = X.drop(train_x.index,axis=0)
    test_y = y[test_x.index]

    return train_x, train_y, test_x, test_y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
