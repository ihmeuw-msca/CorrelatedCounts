# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~

    Utility functions.
"""
import numpy as np


def vec_to_beta(vec, d):
    """Convert vector into correlated model beta structure.

    Parameters
    ----------
    vec : array_like
        Vector that needed to the convert.
    d : array_like
        Number of covariates for each parameter and outcome. Come from
        correlated model.

    Returns
    -------
    :obj: `list` of :obj: `list` of :obj: `numpy.ndarray`
        `beta` structure in the correlated model.
    """
    num_covs = np.sum(d, axis=1)
    pieces = split(vec, num_covs)
    beta = []
    for k, p in enumerate(pieces):
        beta.append(split(p, d[k]))
    return beta


def beta_to_vec(beta):
    """Convert vector into correlated model beta structure.

    Parameters
    ----------
    beta : array_like
        Fixed effects in the correlated model

    Returns
    -------
    array_like:
        Combined vector.
    """
    pieces = []
    for k, p in enumerate(beta):
        pieces.append(np.hstack(p))
    return np.hstack(pieces)


def split(vec, sizes):
    """Split the vector by sizes.

    Parameters
    ----------
    vec : array_like
        Vector that will be break down.
    sizes: array_like
        Size for each piece.

    Returns
    -------
    :obj: `list` of :obj: `numpy.ndarray`
        A list of all the breakdown pieces.
    """
    assert isinstance(vec, np.ndarray)
    assert isinstance(sizes, np.ndarray)
    assert sizes.dtype == int
    assert vec.shape == (np.sum(sizes),)
    return np.split(vec, np.cumsum(sizes)[:-1])