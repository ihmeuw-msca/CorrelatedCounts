# -*- coding: utf-8 -*-
"""
    optimization
    ~~~~~~~~~~~~

    Optimization module for the Correlated Count.
"""
import numpy as np
from . import utils


class OptimizationInterface:
    """Optimization interface for correlated model.
    Current use "EM" algorithm with steps
        * optimize over beta,
        * optimize over random effect U,
        * compute the empirical covariance matrix for U.
    """
    def __init__(self, cm):
        """Optimization interface initialization method.

        Parameters
        ----------
        cm : ccount.core.CorrelatedModel
            Correlated model interface.

        """
        self.cm = cm

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
        return self.cm.log_likelihood(beta=beta)

    def gradient_beta(self, vec, eps=1e-10):
        """Gradient function for fitting the fixed effects.

        Parameters
        ----------
        vec : array_like
            Provided vectorized fixed effects.

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
        return self.cm.log_likelihood(U=U)

    def gradient_U(self, vec, eps=1e-10):
        """Gradient function for fitting the random effects.

        Parameters
        ----------
        vec : array_like
            Provided vectorized random effects.

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
