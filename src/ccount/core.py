# -*- coding: utf-8 -*-
"""
    core
    ~~~~

    Core module for correlated count.
"""
import numpy as np
import numbers


class CorrelatedModel:
    """Correlated model with multiple outcomes.

    Attributes
    ----------
    m : int
        Number of individuals.
    n : int
        Number of outcomes.
    l : int
        Number of parameters in the considered distribution.
    d : array_like
        Number of covariates for each parameter and outcome.
    Y : array_like
        Observations.
    X : :obj: `list` of :obj: `list` of :obj: `numpy.ndarray`
        List of list of 2D arrays, storing the covariates for each parameter
        and outcome.
    g : :obj: `list` of :obj: `function`
        List of link functions for each parameter.
    f : function
        Log likelihood function, better be `numpy.ufunc`.
    beta : :obj: `list` of :obj: `list` of :obj: `numpy.ndarray`
        Fixed effects for predicting the parameters.
    U : array_like
        Random effects for predicting the parameters. Assume random effects
        follow multi-normal distribution.
    D : array_like
        Covariance matrix for the random effects distribution.
    P : array_like
        Parameters for each individual and outcome.

    """

    def __init__(self, m, n, l, d, Y, X, g, f):
        """Correlated Model initialization method.

        Parameters
        ----------
        m : int
            Number of individuals.
        n : int
            Number of outcomes.
        l : int
            Number of parameters in the considered distribution.
        d : array_like
            Number of covariates for each parameter and outcome.
        Y : array_like
            Observations.
        X : :obj: `list` of :obj: `list` of :obj: `numpy.ndarray`
            List of list of 2D arrays, storing the covariates for each parameter
            and outcome.
        g : :obj: `list` of :obj: `function`
            List of link functions for each parameter.
        f : function
            Log likelihood function, better be `numpy.ufunc`.

        """
        # dimension
        self.m = m
        self.n = n
        self.l = l
        self.d = d
        
        # data and covariates
        self.Y = Y
        self.X = X

        # link and log likelihood functions
        self.g = g
        self.f = f

        # check input
        self.check()

        # fixed effects
        self.beta = [[np.zeros(self.d[k, j])
                      for j in range(self.n)] for k in range(self.l)]

        # random effects and its covariance matrix
        self.U = np.zeros((self.l, self.m, self.n))
        self.D = np.array([np.identity(self.n) for k in range(self.l)])

        # place holder for parameter
        self.P = np.zeros((self.l, self.m, self.n))

    def check(self):
        """Check the type, value and size of the inputs."""
        # types
        assert isinstance(self.m, int)
        assert isinstance(self.n, int)
        assert isinstance(self.l, int)
        assert isinstance(self.d, np.ndarray)
        assert self.d.dtype == int

        assert isinstance(self.Y, np.ndarray)
        assert issubclass(self.Y.dtype, numbers.Number)
        assert isinstance(self.X, list)
        for X_k in self.X:
            assert isinstance(X_k, list)
            for X_kj in X_k:
                assert isinstance(X_kj, np.ndarray)
                assert issubclass(X_kj.dtype, numbers.Number)

        assert isinstance(self.g, list)
        assert all(callable(g_k) for g_k in self.g)
        assert callable(self.f)

        # values
        assert self.m > 0
        assert self.n > 0
        assert self.l > 0
        assert np.all(self.d > 0)

        # sizes
        assert self.Y.shape == (self.m, self.n)
        assert len(self.X) == self.l
        assert all(len(self.X[k]) == self.n for k in range(self.l))
        assert all(self.X[k][j].shape == (self.m, self.d[k, j])
                   for k in range(self.l)
                   for j in range(self.n))

        assert len(self.g) == self.l

    def update_params(self, beta=None, U=None, D=None, P=None):
        """Update the variables related to the parameters.

        Parameters
        ----------
        beta : :obj: `list` of :obj: `list` of :obj: `numpy.ndarray`, optional
            Fixed effects for predicting the parameters.
        U : :obj: `numpy.ndarray`, optional
            Random effects for predicting the parameters. Assume random effects
            follow multi-normal distribution.
        D : :obj: `numpy.ndarray`, optional
            Covariance matrix for the random effects distribution.
        P : :obj: `numpy.ndarray`, optional
            Parameters for each individual and outcome. If `P` is provided,
            the `self.P` will be overwrite by its value, otherwise,
            the `self.P` will be updated by the fixed and random effects.

        """
        if beta is not None:
            self.beta = beta
        if U is not None:
            self.U = U
        if D is not None:
            self.D = D
        if P is not None:
            self.P = P
        else:
            P = np.array([self.X[k][j].dot(self.beta[k][j])
                          for k in range(self.l)
                          for j in range(self.n)])
            self.P = P.reshape((self.l, self.n, self.m)).transpose(0, 2, 1)
            self.P += self.U
            for k in range(self.l):
                self.P[k] = self.g[k](self.P[k])

    def log_likelihood(self):
        """Return the log likelihood of the model.

        Returns
        -------
        float
            Average log likelihood.

        """
        return np.mean(self.f(self.Y, self.P))
