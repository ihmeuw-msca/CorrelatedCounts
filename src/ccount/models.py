import numpy as np
from ccount.core import CorrelatedModel
import logging

LOG = logging.getLogger(__name__)


class ZeroInflatedPoisson(CorrelatedModel):
    """
    A Zero-Inflated Poisson Model.

    Example:
    >>> from ccount.simulate import PoissonSimulation
    >>> ps = PoissonSimulation(m=100, n=2, d=[2, 2])
    >>> Y = ps.simulate()
    >>> X = [[np.ones((ps.m, 1)) for i in range(ps.n)], ps.x]
    >>> D = np.array([[1, 1], ps.d])
    >>> zp = ZeroInflatedPoisson(m=ps.m, d=D, Y=Y, X=X)
    >>> zp.optimize_params()
    """
    def __init__(self, m, d, Y, X):
        LOG.info("Initializing a Zero-Inflated Poisson Model")
        assert len(d) == 2
        assert len(X) == 2
        super().__init__(
            m=m, n=2, d=d, Y=Y.astype(np.number), X=X,
            l=2, g=[lambda x: np.exp(x) / (1 + np.exp(x)), np.exp],
            f=zip_neg_log_likelihood
        )


def zip_neg_log_likelihood(Y, P):
    """
    The negative log likelihood for
    zero-inflated Poisson.

    Parameters
    ----------
    Y : array_like
    P : list
    """
    p = P[0]
    theta = P[1]
    likelihood = ((p + (1 - p) * np.exp(-theta)) * (Y == 0) +
                  ((1 - p) * np.exp(-theta) * theta ** Y) * (1 - (Y == 0)))
    return -np.log(likelihood)
