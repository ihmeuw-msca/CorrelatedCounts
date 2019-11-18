import numpy as np


class Simulation:
    def __init__(self, m, n, d):
        """
        Simulation setup for creating correlated count data.
        There are m individuals, n correlated count bins, and d[n] fixed effects
        for each n.

        Parameters
        -----------
            m : int
                Number of individuals in the simulation
            n : int
                Number of categories for the correlated counts
            d : array_like
                Number of covariates for each parameter and outcome

        """
        # Dimension parameters
        assert isinstance(m, int)
        assert isinstance(n, int)
        assert isinstance(d, list)
        if len(d) != n:
            raise ValueError("d_k needs to be list of length k")
        self.m = m
        self.n = n
        self.d = d


class PoissonSimulation(Simulation):
    """
    Simulate data from a Correlated Poisson

    Example
    ----------
    >>> s = PoissonSimulation(m=100, n=2, d=[2, 2])
    >>> s.simulate()

    >>> s.update_params(p=0.2)
    >>> s.simulate()

    >>> s.x[0] = np.ones((s.m, 1)) # get rid of covariates for the first parameter
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Baseline simulation parameters
        self.p = 0.5
        self.D = np.identity(n=self.n)
        self.x = [np.random.randn(self.m, self.d[j]) for j in range(self.n)]
        self.beta = [np.random.randn(self.d[j]) for j in range(self.n)]

        # Simulation results
        self.u = None
        self.theta = None

        self.Y_zeros = None
        self.Y_poisson = None
        self.Y_zip = None

    def update_params(self, x=None, beta=None, p=None, D=None):
        """
        Update the parameters from baseline for the simulation

        Parameters
        ----------
        p : float
            Probability of observing a 0
        x : List[np.ndarray]
            List of length J of np.ndarray with dimension (n, d[j])
        beta : List[np.ndarray]
            List of length J with np.arrays of dimension d[j]
        D : np.ndarray
            2D variance-covariance matrix of square dimension J for the random effects
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

        Attributes
        ----------
        self.u : np.ndarray
            2D array of shape (m, n), m random realizations
            of the random effect u where
            .. math::
                u \sim N(0, D)
        self.theta : List[np.ndarray]
            list of length J where
            .. math::
                theta_{j} = e^{x_{j} beta_{j} + u_{j}^T}
        self.y_zeros : np.ndarray
            2D array of shape (n, m) that induces
            structural zeroes
        self.y_poisson : np.ndarray
            2D array of shape (n, m) with poisson
            realizations with mean theta_sim
            .. math::
                y_poisson_{i, j} \sim Poisson(lambda=theta{i, j})
        self.y_zip : np.ndarray
            2D array of shape (n, m) with poisson
            realizations masked by the structural zeros from p_sim
        """
        self.u = np.random.multivariate_normal(size=self.m,
                                               mean=np.zeros(self.n),
                                               cov=self.D)
        self.theta = [np.exp(self.x[j].dot(self.beta[j]) + self.u.T[j])
                      for j in range(self.n)]

        self.Y_zeros = 1 - np.random.binomial(size=(self.n, self.m),
                                              p=self.p, n=1)
        self.Y_poisson = np.random.poisson(lam=self.theta)
        self.Y_zip = self.Y_zeros * self.Y_poisson

        return self.Y_zip.T
