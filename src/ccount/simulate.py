import numpy as np
from numpy.random import multivariate_normal
from numpy import identity


class Simulation:
    def __init__(self, n=100, k=2, p=0.5, beta=None, d=None):
        """
        Simulation setup for creating correlated count data for mortality and incidence.

        :param n: number of individuals in the simulation
        :param k: number of categories for the correlated counts
        :param p: probability of observing a 0
        :param beta: fixed effects
        :param d: variance-covariance matrix for the random effects
        """
        self.n = n
        self.k = k
        self.p = p
        if beta is not None:
            self.beta = beta
        else:
            self.beta = np.array([])
        if d is not None:
            self.d = d
        else:
            self.d = identity(n=self.n)

        self.x = None
        self.y = None
        self.u = None

    def simulate(self):
        self.u = multivariate_normal(size=self.n, mean=np.zeros(self.k), cov=self.d)

