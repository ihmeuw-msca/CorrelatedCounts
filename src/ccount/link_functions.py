import numpy as np


def expit(x):
    return np.exp(x) / (1 + np.exp(x))


def smooth_ReLU(x, x_limit=50):
    above_limit = x > x_limit
    return above_limit * x + ~above_limit * np.log(1 + np.exp(x))
