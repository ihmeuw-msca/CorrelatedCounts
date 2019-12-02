# -*- coding: utf-8 -*-
"""
    test_core
    ~~~~~~~~~

    Test the core module
"""
import numpy as np
import pytest
import ccount.core as core
import ccount.utils as utils

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
                              lambda y, p: 0.5*(y - p[0])**2)
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
@pytest.mark.parametrize("U", [None, np.ones((l, m, n))])
def test_correlated_model_compute_P(beta, U):
    cm = core.CorrelatedModel(m, n, l, d, Y, X,
                              [lambda x: x] * l,
                              lambda y, p: 0.5 * (y - p[0]) ** 2)
    P = cm.compute_P(beta=beta, U=U)
    if beta is None:
        beta = cm.beta
    if U is None:
        U = cm.U
    true_P = np.array([cm.X[k][j].dot(beta[k][j])
                  for k in range(l)
                  for j in range(n)])
    true_P = true_P.reshape((l, n, m)).transpose(0, 2, 1) + U

    assert np.linalg.norm(P - true_P) < 1e-10


@pytest.mark.parametrize("beta",
                         [None, [[np.ones(d[k, j])
                                  for j in range(n)] for k in range(l)]])
@pytest.mark.parametrize("U", [None, np.zeros((l, m, n))])
@pytest.mark.parametrize("D", [None,
                               np.array([np.identity(n) for i in range(l)])])
@pytest.mark.parametrize("P", [None, np.random.randn(l, m, n)])
def test_correlated_model_update_params(beta, U, D, P):
    cm = core.CorrelatedModel(m, n, l, d, Y, X,
                              [lambda x: x] * l,
                              lambda y, p: 0.5*(y - p[0])**2)

    cm.update_params(beta=beta, U=U, D=D, P=P)
    if beta is None:
        beta = cm.beta
    assert np.linalg.norm(utils.beta_to_vec(beta) -
                          utils.beta_to_vec(cm.beta)) < 1e-10
    if U is None:
        U = cm.U
    assert np.linalg.norm(U - cm.U) < 1e-10
    if D is None:
        D = cm.D
    assert np.linalg.norm(D - cm.D) < 1e-10
    if P is None:
        P = cm.P
    assert np.linalg.norm(P - cm.P) < 1e-10


@pytest.mark.parametrize("beta",
                         [None, [[np.ones(d[k, j])
                                  for j in range(n)] for k in range(l)]])
@pytest.mark.parametrize("U", [None, np.zeros((l, m, n))])
@pytest.mark.parametrize("D", [None,
                               np.array([np.identity(n) for i in range(l)])])
def test_correlated_model_neg_log_likelihood(beta, U, D):
    cm = core.CorrelatedModel(m, n, l, d, Y, X,
                              [lambda x: x] * l,
                              lambda y, p: 0.5*(y - p[0])**2)

    cm.update_params(beta=beta, U=U, D=D)
    assert np.abs(cm.neg_log_likelihood() -
                  0.5*np.mean(np.sum((cm.Y - cm.P[0])**2, axis=1)) -
                  0.5*np.sum(cm.U[0]*cm.U[0])/cm.m) < 1e-10
