import numpy as np
import logging

from ccount.core import CorrelatedModel
from ccount.likelihoods import NegLogLikelihoods
from ccount.link_functions import smooth_ReLU, expit

LOG = logging.getLogger(__name__)


class HurdlePoisson(CorrelatedModel):
    """
    A Hurdle Model. Has a binomial model
    for the proportion of zeros, and a zero-truncated
    Poisson for the
    """
    def __init__(self, m, n, d, Y, X, group_id=None, offset=None):
        LOG.info("Initializing a Hurdle Poisson Model.")
        assert len(d) == 2
        assert len(X) == 2
        super().__init__(
            m=m, n=n, d=d, Y=Y.astype(np.number), X=X, group_id=group_id, offset=offset,
            l=2, g=[expit, np.exp],
            f=NegLogLikelihoods.hurdle_poisson
        )
        self.model_type = "Hurdle Poisson"
        self.parameters = [
            "Probability of Zero", "Mean of Poisson"
        ]

    @staticmethod
    def mean_outcome(P):
        p = P[0]
        theta = P[1]
        return (1 - p) * theta / (1 - np.exp(-theta))


class ZeroInflatedPoisson(CorrelatedModel):
    """
    A Zero-Inflated Poisson Model.

    Example:
    >>> from ccount.simulate import ZIPoissonSimulation
    >>> ps = ZIPoissonSimulation(m=100, n=2, d=[2, 2])
    >>> Y = ps.simulate()
    >>> X = [[np.ones((ps.m, 1)) for i in range(ps.n)], ps.x]
    >>> D = np.array([[1, 1], ps.d])
    >>> zp = ZeroInflatedPoisson(m=ps.m, d=D, Y=Y, X=X)
    >>> zp.optimize_params()
    """
    def __init__(self, m, n, d, Y, X, group_id=None, offset=None, weights=None,
                 add_intercepts=True, normalize_X=True):
        LOG.info("Initializing a Zero-Inflated Poisson Model")
        assert len(d) == 2
        assert len(X) == 2
        super().__init__(
            m=m, n=n, d=d, Y=Y.astype(np.number), X=X,
            group_id=group_id, offset=offset, weights=weights,
            normalize_X=normalize_X, add_intercepts=add_intercepts,
            l=2, g=[lambda x: np.exp(x) / (1 + np.exp(x)), np.exp],
            f=NegLogLikelihoods.zi_poisson
        )
        self.model_type = "Zero-Inflated Poisson"
        self.parameters = [
            "Probability of Structural Zero", "Mean of Poisson"
        ]

    @staticmethod
    def mean_outcome(P):
        p = P[0]
        theta = P[1]
        return (1 - p) * theta


class ZeroInflatedPoissonSmoothReLU(CorrelatedModel):
    """
    A Zero-Inflated Poisson likelihood with a Smooth ReLU link function
    rather than a log link for the Poisson mean.
    """
    def __init__(self, m, n, d, Y, X, group_id=None, offset=None, weights=None,
                 add_intercepts=True, normalize_X=True):
        LOG.info("Initializing a Zero-Inflated Poisson SmoothReLU Model")
        assert len(d) == 2
        assert len(X) == 2
        super().__init__(
            m=m, n=n, d=d, Y=Y.astype(np.number), X=X,
            group_id=group_id, offset=offset, weights=weights,
            add_intercepts=add_intercepts, normalize_X=normalize_X,
            l=2, g=[expit, smooth_ReLU],
            f=NegLogLikelihoods.zi_poisson
        )
        self.model_type = "Zero-Inflated Poisson Smooth ReLU"
        self.parameters = [
            "Probability of Structural Zero", "Mean of Poisson"
        ]


class NegativeBinomial(CorrelatedModel):
    """
    A Negative Binomial Model.
    """
    def __init__(self, m, n, d, Y, X, group_id=None, offset=None, weights=None,
                 add_intercepts=True, normalize_X=True):
        LOG.info("Initializing a negative binomial model.")
        assert len(d) == 2
        assert len(X) == 2
        super().__init__(
            m=m, n=n, d=d, Y=Y.astype(np.number), X=X, group_id=group_id, offset=offset,
            add_intercepts=add_intercepts, normalize_X=normalize_X, weights=weights,
            l=2, g=[np.exp, np.exp],
            f=NegLogLikelihoods.nbinom
        )
        self.model_type = "Negative Binomial"
        self.parameters = [
            "Mean of Poisson", "Over-Dispersion Parameter Variance"
        ]

    @staticmethod
    def mean_outcome(P):
        theta = P[0]
        k = P[1]  # The over-dispersion parameter is not used for the mean value calculation
        return theta


MODEL_DICT = {
    'hurdle_poisson': HurdlePoisson,
    'negative_binomial': NegativeBinomial,
    'zero_inflated_poisson': ZeroInflatedPoisson,
    'zero_inflated_poisson_relu': ZeroInflatedPoissonSmoothReLU
}
