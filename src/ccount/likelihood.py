import numpy as np
from scipy.stats import multivariate_normal


def theta(x, b, u):
    """
    Expression for \theta: \theta = exp(x_{i, j}^T \beta_j + u_{i, j}

    :param x: (np.array) a vector of observed covariates
    :param b: (np.array) a vector of parameters
    :param u: (float) the random effect
    :return: (float) value of theta
    """
    return -np.exp(np.matmul(x, b) + u)


def zero_inflated_poisson(p, th, y):
    """

    :param p: (float) probability of being a 0
    :param th: (float) theta parameter
    :param y: (int) the count outcome
    :return: (int)
    """
    ind = y == 0
    return ((p + (1 - p) * np.exp(-th)) * ind +
            ((1 - p) * np.exp(-th) * th ** y / np.math.factorial(y)) * (1 - ind))


def joint_likelihood(p, b, u, y, x, model_type='Poisson'):
    """
    The joint likelihood for the data.

    :param p: (float) p for the zero-inflated Poisson
    :param b: (np.array) vector of floats, the coefficients on x
    :param u: (np.ndarray) 2D array of random effects U
    :param y: (np.ndarray) 2D array of count data
    :param x: (np.ndarray) 3D array of covariates for prediction
    :param model_type: (str) type of likelihood. Right now only 'Poisson' implemented.
        In the future, might have 'NegativeBinomial'
    :return:
    """
    assert 0 <= p <= 1
    assert type(y) is int
    assert len(x.shape) == 3
    assert x.shape[0] == 2
    assert len(y.shape) == 2
    assert y.shape[1]

    if model_type == 'Poisson':
        joint = np.product([
            zero_inflated_poisson(
                p=p,
                th=theta(x[i][j], b[j], u[i]),
                y=y[i][j]
            ) for i in y.shape[0] for j in y.shape[1]
        ])
    else:
        raise NotImplementedError("Do not have any types besides Poisson implemented.")

    return joint


def u_prior_density(u, d):
    """
    Prior on the u random effect.
    
    :param u: (np.array) array of random effects u
    :param d: (np.ndarray) 2D covariance matrix for multivariate normal distribution for U
    :return:
    """
    multivariate_normal.pdf(u, mean=np.zeros(2), cov=d)
    return None
