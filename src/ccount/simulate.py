import numpy as np
from numpy.random import multivariate_normal, binomial
from numpy import identity


class Simulation:
    def __init__(self, n, J, d):
        """
        Simulation setup for creating correlated count data for mortality and incidence.

        There are n individuals, k correlated count bins, and d_k fixed effects.

        :param n: (int) number of individuals in the simulation
        :param J: (int) number of categories for the correlated counts
        :param d: (List[int]) list of integers (of length J) representing the length of the
            fixed effects vector for each of the J categories

        Usage:
        >>> s = Simulation(n=100, k=2, d_k=[2, 2])
        >>> s.update_params()
        """
        # Dimension parameters
        assert type(n) is int
        assert type(J) is int
        assert type(d) is list
        if len(d) != J:
            raise ValueError("d_k needs to be list of length k")
        self.n = n
        self.J = J
        self.d = d

        # Baseline simulation parameters
        self.p = 0.5
        self.D = identity(n=self.J)
        self.x = [multivariate_normal(mean=np.zeros(d_j), cov=identity(n=d_j), size=self.n) for d_j in self.d]
        self.beta = [multivariate_normal(mean=np.zeros(d_j), cov=identity(n=d_j), size=1) for d_j in self.d]

        # Simulation results
        self.p_sim = None
        self.y_sim = None
        self.u_sim = None

    def update_params(self, x=None, beta=None, p=None, D=None):
        """
        Update the parameters from baseline for the simulation

        :param p: (int) probability of observing a 0
        :param x: (List[np.ndarray]) list of length J of np.ndarray with dimension (n, d[j])
        :param beta: (List[np.array]) list of length J with np.arrays of dimension d[j]
        :param D: (np.ndarray) 2D variance-covariance matrix of square dimension J for the random effects
        """
        self.x = x if x else self.x
        self.beta = beta if beta else self.beta
        self.p = p if p else self.p
        self.D = D if D else self.D

    def simulate(self):
        """
        Make one simulation of the outcome y based on current parameters.
        :return:
        """
        self.p_sim = binomial(size=self.n, p=self.p, n=1)
        self.u_sim = multivariate_normal(size=self.n, mean=np.zeros(self.J), cov=self.D)
        