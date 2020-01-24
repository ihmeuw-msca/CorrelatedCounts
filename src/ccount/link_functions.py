import numpy as np


def expit(x):
    return np.exp(x) / (1 + np.exp(x))


def smooth_ReLU(x, x_limit=50):
    if type(x) != np.array:
        x = np.asarray(x)
    above_limit = x > x_limit
    result = np.log(1 + np.exp(x))
    result[above_limit] = x[above_limit]
    return result
