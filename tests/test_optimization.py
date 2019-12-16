# -*- coding: utf-8 -*-
"""
    test_optimization
    ~~~~~~~~~~~~~~~~~

    Test optimization module for the Correlated Count model.
"""
import numpy as np
import pytest
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
    group_id = None
    cm = core.CorrelatedModel(m, n, l, d, Y, X,
                              [lambda x: x] * l,
                              lambda y, p: 0.5*(y - p[0])**2,
                              group_id=group_id)
    return cm


@pytest.mark.parametrize("beta",
                         [None, [[np.ones(d[k, j])
                                  for j in range(n)] for k in range(l)]])
def test_optimization_objective_beta(cm, beta):
    opt = cm.opt_interface
    if beta is None:
        beta = cm.beta
    vec = utils.beta_to_vec(beta)
    assert np.abs(cm.neg_log_likelihood(beta=beta) -
                  opt.objective_beta(vec)) < 1e-10


@pytest.mark.parametrize("beta",
                         [None, [[np.ones(d[k, j])
                                  for j in range(n)] for k in range(l)]])
def test_optimization_gradient_beta(cm, beta):
    opt = cm.opt_interface
    if beta is None:
        beta = cm.beta
    vec = utils.beta_to_vec(beta)
    full_U = np.repeat(cm.U, cm.group_sizes, axis=1)

    assert np.linalg.norm(opt.gradient_beta(vec) -
                          cm.X[0][0].T.dot(
                              cm.X[0][0].dot(beta[0][0]) + full_U[0].T[0] -
                              cm.Y.T[0])/cm.m) < 1e-8


@pytest.mark.parametrize("U", [None, np.ones((l, m, n))])
def test_optimization_objective_U(cm, U):
    opt = cm.opt_interface
    if U is None:
        U = cm.U
    assert np.abs(opt.objective_U(U.flatten()) -
                  cm.neg_log_likelihood(U=U)) < 1e-10


@pytest.mark.parametrize("U", [None, np.ones((l, m, n))])
def test_optimization_gradient_U(cm, U):
    opt = cm.opt_interface
    if U is None:
        U = cm.U
    full_U = np.repeat(U, cm.group_sizes, axis=1)

    my_grad = opt.gradient_U(U.flatten()).reshape(cm.U.shape)
    tr_grad = (cm.X[0][0].dot(cm.beta[0][0]) + full_U.flatten() -
               cm.Y.T[0])/cm.m
    tr_grad = tr_grad.reshape(full_U.shape)
    tr_grad = np.add.reduceat(tr_grad,
                              np.cumsum([0] + list(cm.group_sizes))[:-1],
                              axis=1)
    tr_grad += U/cm.num_groups
    assert np.linalg.norm(my_grad - tr_grad) < 1e-10


def test_optimization_optimize_beta(cm):
    mat = cm.X[0][0]
    full_U = np.repeat(cm.U, cm.group_sizes, axis=1)
    vec = cm.Y.T[0] - full_U.flatten()
    true_beta = np.linalg.solve(mat.T.dot(mat), mat.T.dot(vec))
    cm.opt_interface.optimize_beta()
    assert np.linalg.norm(true_beta - utils.beta_to_vec(cm.beta)) < 1e-5


def test_optimization_optimize_U(cm):
    true_U = 0.5*(cm.Y.T[0] - cm.X[0][0].dot(cm.beta[0][0]))
    cm.opt_interface.optimize_U()
    full_U = np.repeat(cm.U, cm.group_sizes, axis=1)
    assert np.linalg.norm(true_U - full_U.flatten()) < 1e-5


def test_optimization_compute_D(cm):
    cm.opt_interface.optimize_U()
    cm.opt_interface.compute_D()
    assert np.linalg.norm(cm.D - np.cov(cm.U.flatten())) < 1e-8
