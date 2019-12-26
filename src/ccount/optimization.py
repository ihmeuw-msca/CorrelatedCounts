# -*- coding: utf-8 -*-
"""
    optimization
    ~~~~~~~~~~~~

    Optimization module for the Correlated Count.
"""
import numpy as np
import scipy.optimize as sopt
from . import utils
import logging

LOG = logging.getLogger(__name__)


class OptimizationInterface:
    """Optimization interface for correlated model.
    Current use "EM" algorithm with steps
        * optimize over beta,
        * optimize over random effect U,
        * compute the empirical covariance matrix for U.
    """
    def __init__(self, cm, n_iteration_print=10):
        """Optimization interface initialization method.

        Parameters
        ----------
        cm : ccount.core.CorrelatedModel
            Correlated model interface.
        n_iteration_print : int
            Will print the objective function value every n_iteration_print iters

        """
        self.cm = cm
        self.EVALUATIONS = 0
        self.n_iteration_print = n_iteration_print

    def objective_beta(self, vec):
        """Objective function for fitting the fixed effects.

        Parameters
        ----------
        vec : array_like
            Provided vectorized fixed effects.

        Returns
        -------
        float
            Objective function value.
        """
        beta = utils.vec_to_beta(vec, self.cm.d)
        return self.cm.neg_log_likelihood(beta=beta)

    def gradient_beta(self, vec, eps=1e-10):
        """Gradient function for fitting the fixed effects.

        Parameters
        ----------
        vec : array_like
            Provided vectorized fixed effects.
        eps : float

        Returns
        -------
        numpy.ndarray
            Gradient at current fixed effects.
        """
        g_vec = np.zeros(vec.size)
        c_vec = vec + 0j
        for i in range(vec.size):
            c_vec[i] += eps*1j
            g_vec[i] = self.objective_beta(c_vec).imag/eps
            c_vec[i] -= eps*1j

        return g_vec

    def objective_U(self, vec):
        """Objective function for fitting the random effects.

        Parameters
        ----------
        vec : array_like
            Provided vectorized random effects.

        Returns
        -------
        float
            Objective function value.
        """
        U = vec.reshape(self.cm.U.shape)
        return self.cm.neg_log_likelihood(U=U)

    def gradient_U(self, vec, eps=1e-10):
        """Gradient function for fitting the random effects.

        Parameters
        ----------
        vec : array_like
            Provided vectorized random effects.
        eps : float

        Returns
        -------
        numpy.ndarray
            Gradient at current random effects.
        """
        g_vec = np.zeros(vec.size)
        c_vec = vec + 0j
        for i in range(vec.size):
            c_vec[i] += eps*1j
            g_vec[i] = self.objective_U(c_vec).imag/eps
            c_vec[i] -= eps*1j

        return g_vec

    def optimize_beta(self):
        """Optimize fixed effects.
        """
        LOG.info("Optimizing beta.")
        self.EVALUATIONS = 1
        print('{0:4s}    {1:9s}'.format('Iteration', 'Objective Function Value'))
        result = sopt.minimize(self.objective_beta,
                               utils.beta_to_vec(self.cm.beta),
                               jac=self.gradient_beta,
                               method="L-BFGS-B",
                               callback=self.callback_beta)
        self.cm.update_params(beta=utils.vec_to_beta(result.x, self.cm.d))

    def optimize_U(self):
        """Optimize random effects.
        """
        LOG.info("Optimizing U.")
        self.EVALUATIONS = 1
        print('{0:4s}    {1:9s}'.format('Iteration', 'Objective Function Value'))
        result = sopt.minimize(self.objective_U,
                               self.cm.U.flatten(),
                               jac=self.gradient_U,
                               method="L-BFGS-B",
                               callback=self.callback_U)
        self.cm.update_params(U=result.x.reshape(self.cm.U.shape))

    def compute_D(self):
        """Compute the sample covariance of the random effects.
        """
        LOG.info("Computing D.")
        D = np.array([np.cov(self.cm.U[k].T) for k in range(self.cm.l)])
        self.cm.update_params(D=D)

    def callback_beta(self, X):
        if self.EVALUATIONS % 10 == 0:
            print('{0:4d}        {1: 3.6f}'.format(
                self.EVALUATIONS, self.objective_beta(X))
            )
        self.EVALUATIONS += 1

    def callback_U(self, X):
        if self.EVALUATIONS % 10 == 0:
            print('{0:4d}        {1: 3.6f}'.format(
                self.EVALUATIONS, self.objective_U(X))
            )
        self.EVALUATIONS += 1
