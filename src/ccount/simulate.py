import numpy as np


class Simulation:
    def __init__(self, n, J, d):
        """
        Simulation setup for creating correlated count data.
        There are n individuals, J correlated count bins, and d[j] fixed effects
        for each J.

        Parameters:
            n: (int) number of individuals in the simulation
            J: (int) number of categories for the correlated counts
            d: (List[int]) list of integers (of length J) representing the
                length of the fixed effects vector for each of the J categories

        Usage:
        >>> s = Simulation(n=100, J=2, d=[2, 2])
        >>> s.simulate()

        >>> s.update_params(p=0.2)
        >>> s.simulate()
        """
        # Dimension parameters
        assert isinstance(n, int)
        assert isinstance(J, int)
        assert isinstance(d, list)
        if len(d) != J:
            raise ValueError("d_k needs to be list of length k")
        self.n = n
        self.J = J
        self.d = d

        # Baseline simulation parameters
        self.p = 0.5
        self.D = np.identity(n=self.J)
        self.x = [np.random.randn(self.n, self.d[j]) for j in range(self.J)]
        self.beta = [np.random.randn(self.d[j]) for j in range(self.J)]

        # Simulation results
        self.u = None
        self.theta = None

        self.y_zeros = None
        self.y_poisson = None
        self.y_zip = None

    def update_params(self, x=None, beta=None, p=None, D=None):
        """
        Update the parameters from baseline for the simulation

        Parameters:
            p: (int) probability of observing a 0
            x: (List[np.ndarray]) list of length J of np.ndarray with dimension
                (n, d[j])
            beta: (List[np.ndarray]) list of length J with np.arrays of
                dimension d[j]
            D: (np.ndarray) 2D variance-covariance matrix of square dimension J
                for the random effects
        """
        if x is not None:
            self.x = x
        if beta is not None:
            self.beta = beta
        if p is not None:
            self.p = p
        if D is not None:
            self.D = D

    def simulate(self):
        """
        Make one simulation of the outcome y based on current parameters.

        Attributes:
            self.u: (np.ndarray) 2D array of shape (n, J), n random realizations
                of the random effect u where
                .. math::
                    u \sim N(0, D)
            self.theta: (List[np.ndarray]) list of length J where
                .. math::
                    theta_{j} = e^{x_{j} beta_{j} + u_{j}^T}
            self.y_zeros: (np.ndarray) 2D array of shape (J, n) that induces
                structural zeroes
            self.y_poisson: (np.ndarray) 2D array of shape (J, n) with poisson
                realizations with mean theta_sim
                .. math::
                    y_poisson_{i, j} \sim Poisson(lambda=theta{i, j})
            self.y_zip: (np.ndarray) 2D array of shape (J, n) with poisson
                realizations masked by the structural zeros from p_sim
        """
        self.u = np.random.multivariate_normal(size=self.n,
                                               mean=np.zeros(self.J),
                                               cov=self.D)
        self.theta = [np.exp(self.x[j].dot(self.beta[j]) + self.u.T[j])
                      for j in range(self.J)]

        self.y_zeros = 1 - np.random.binomial(size=(self.J, self.n),
                                              p=self.p, n=1)
        self.y_poisson = np.random.poisson(lam=self.theta)
        self.y_zip = self.y_zeros * self.y_poisson

        return self.y_zip
