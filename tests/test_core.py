# -*- coding: utf-8 -*-
"""
    test_core
    ~~~~~~~~~

    Test the core module
"""
import numpy as np
import pytest
import ccount.core as core

# test problem
m = 5
n = 3
l = 1
d = np.array([[2]*n]*l)

Y = np.random.randn(m, n)
X = [[np.random.randn(m, d[k, j])
      for j in range(n)] for k in range(l)]


def test_correlated_model():
    cm = core.CorrelatedModel(m, n, l, d, Y, X,
                              [lambda x: x] * l,
                              lambda y, p: 0.5*(y - p)**2)
    assert all([np.linalg.norm(cm.beta[k][j]) < 1e-10
                for k in range(l)
                for j in range(n)])
    assert np.linalg.norm(cm.U) < 1e-10
    assert all([np.linalg.norm(cm.D[k] - np.identity(n)) < 1e-10
                for k in range(l)])
    assert np.linalg.norm(cm.P) < 1e-10


@pytest.mark.parametrize("beta",
                         [None, [[np.ones(d[k, j])
                                  for j in range(n)] for k in range(l)]])
@pytest.mark.parametrize("U", [None, np.zeros((l, m, n))])
@pytest.mark.parametrize("D", [None, [np.identity(n) for i in range(l)]])
@pytest.mark.parametrize("P", [None, np.random.randn(l, m, n)])
def test_correlated_model_update_params(beta, U, D, P):
    cm = core.CorrelatedModel(m, n, l, d, Y, X,
                              [lambda x: x] * l,
                              lambda y, p: 0.5*(y - p)**2)

    cm.update_params(beta=beta, U=U, D=D, P=P)
    if P is None:
        P = np.array([cm.X[k][j].dot(cm.beta[k][j])
                      for k in range(l)
                      for j in range(n)])
        P = P.reshape((l, n, m)).transpose(0, 2, 1) + cm.U
    assert np.linalg.norm(cm.P - P) < 1e-10


@pytest.mark.parametrize("beta",
                         [None, [[np.ones(d[k, j])
                                  for j in range(n)] for k in range(l)]])
@pytest.mark.parametrize("U", [None, np.zeros((l, m, n))])
@pytest.mark.parametrize("D", [None, [np.identity(n) for i in range(l)]])
@pytest.mark.parametrize("P", [None, np.random.randn(l, m, n)])
def test_correlated_model_log_likelihood(beta, U, D, P):
    cm = core.CorrelatedModel(m, n, l, d, Y, X,
                              [lambda x: x] * l,
                              lambda y, p: 0.5*(y - p)**2)

    cm.update_params(beta=beta, U=U, D=D, P=P)
    assert np.abs(cm.log_likelihood() - 0.5*np.mean((cm.Y - cm.P)**2)) < 1e-10
