from numpy.random import multivariate_normal, binomial, poisson
from numpy import identity, exp, matmul, zeros, transpose as t


class Simulation:
    def __init__(self, n, J, d):
        """
        Simulation setup for creating correlated count data.
        There are n individuals, J correlated count bins, and d[j] fixed effects for each J.

        Parameters:
            n: (int) number of individuals in the simulation
            J: (int) number of categories for the correlated counts
            d: (List[int]) list of integers (of length J) representing the length of the
                fixed effects vector for each of the J categories

        Usage:
        >>> s = Simulation(n=100, k=2, d=[2, 2])
        >>> s.simulate()

        >>> s.update_params(p=0.2)
        >>> s.simulate()
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
        self.x = [multivariate_normal(mean=zeros(d_j), cov=identity(n=d_j), size=self.n) for d_j in self.d]
        self.beta = [multivariate_normal(mean=zeros(d_j), cov=identity(n=d_j), size=1) for d_j in self.d]

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
            x: (List[np.ndarray]) list of length J of np.ndarray with dimension (n, d[j])
            beta: (List[np.array]) list of length J with np.arrays of dimension d[j]
            D: (np.ndarray) 2D variance-covariance matrix of square dimension J for the random effects
        """
        self.x = x if x else self.x
        self.beta = beta if beta else self.beta
        self.p = p if p else self.p
        self.D = D if D else self.D

    def simulate(self):
        """
        Make one simulation of the outcome y based on current parameters.

        Attributes:
            self.u: (np.ndarray) 2D array of shape (n, J), n random realizations of the random effect u where
                .. math::
                    u \sim N(0, D)
            self.theta: (List[np.ndarray]) list of length J where
                .. math::
                    theta_{j} = e^{x_{j} beta_{j}^T + u_{j}^T}
            self.y_zeros: (np.ndarray) 2D array of shape (J, n) that induces structural zeroes
            self.y_poisson: (np.ndarray) 2D array of shape (J, n) with poisson realizations with mean theta_sim
                .. math::
                    y_poisson_{i, j} \sim Poisson(lambda=theta{i, j})
            self.z_zip: (np.ndarray) 2D array of shape (J, n) with poisson realizations masked
                by the structural zeros from p_sim
        """
        self.u = multivariate_normal(size=self.n, mean=zeros(self.J), cov=self.D)
        self.theta = [exp(
            matmul(self.x[j], t(self.beta[j])).flatten() + t(self.u)[j]
        ) for j in range(self.J)]

        self.y_zeros = 1 - binomial(size=(self.J, self.n), p=self.p, n=1)
        self.y_poisson = poisson(lam=self.theta)
        self.y_zip = self.y_zeros * self.y_poisson

        return self.y_zip
