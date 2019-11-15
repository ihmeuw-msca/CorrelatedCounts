# -*- coding: utf-8 -*-
"""
    test_optimization
    ~~~~~~~~~~~~~~~~~

    Test optimization module for the Correlated Count model.
"""
import numpy as np
import pytest
import ccount.optimization as optimization
import ccount.core as core
import ccount.utils as utils


# dimension settings
m = 5
n = 1
l = 1
d = np.array([[2] * n] * l)

@pytest.fixture()
def cm():
    Y = np.random.randn(m, n)
    X = [[np.random.randn(m, d[k, j])
          for j in range(n)] for k in range(l)]
    cm = core.CorrelatedModel(m, n, l, d, Y, X,
                              [lambda x: x] * l,
                              lambda y, p: 0.5*(y - p)**2)
    return cm


@pytest.mark.parametrize("beta",
                         [None, [[np.ones(d[k, j])
                                  for j in range(n)] for k in range(l)]])
def test_optimization_objective_beta(cm, beta):
    opt = optimization.OptimizationInterface(cm)
    if beta is None:
        beta = cm.beta
    vec = utils.beta_to_vec(beta)
    assert np.abs(cm.log_likelihood(beta=beta) -
                  opt.objective_beta(vec)) < 1e-10


@pytest.mark.parametrize("beta",
                         [None, [[np.ones(d[k, j])
                                  for j in range(n)] for k in range(l)]])
def test_optimization_gradient_beta(cm, beta):
    opt = optimization.OptimizationInterface(cm)
    if beta is None:
        beta = cm.beta
    vec = utils.beta_to_vec(beta)

    assert np.linalg.norm(opt.gradient_beta(vec) -
                          cm.X[0][0].T.dot(
                              cm.X[0][0].dot(beta[0][0]) -
                              cm.Y.T[0])/cm.m) < 1e-8
