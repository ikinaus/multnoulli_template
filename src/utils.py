import numpy as np


def add_intercept_column(x):
    ones = np.ones((x.shape[0], 1))
    return np.hstack((ones, x))


def softmax(Z):
    max_Z = np.max(Z, axis=1, keepdims=True)

    exp_Z = np.exp(Z - max_Z)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    return exp_Z / sum_exp_Z
