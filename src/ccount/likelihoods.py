import numpy as np
from scipy.special import loggamma


class NegLogLikelihoods:

    @staticmethod
    def hurdle_poisson(Y, P):
        """
        Hurdle Poisson likelihood.
        Structural Zeroes induced by binomial distribution, then Non-Zeroes induced
        by truncated Poisson model.

        Args:
            Y: observed data
            P: list with the following elements:
                0: the probability of a zero
                1: mean of the Poisson distribution
        """
        assert P.shape[0] == 2
        p = P[0]
        theta = P[1]
        ll = (
            (np.log(p)) * (Y == 0) +
            (np.log(1 - p) - theta + Y * np.log(theta) - np.log(1 - np.exp(-theta))) * (Y > 0)
        )
        return -ll

    @staticmethod
    def zi_poisson(Y, P):
        """
        Zero-Inflated Poisson likelihood.
        Structural Zeroes induced by either binomial distribution, additional zeroes
        from the Poisson distribution.

        Args:
            Y: observed data
            P: list with the following elements:
                0: the probability of a structural zero
                1: mean of the Poisson distribution

        Returns:
            negative log likelihood
        """
        assert P.shape[0] == 2
        p = P[0]
        theta = P[1]
        ll = (
            (np.log((p + (1 - p) * np.exp(-theta)))) * (Y == 0) +
            (np.log(1 - p) - theta + Y * np.log(theta)) * (Y > 0)
        )
        return -ll

    @staticmethod
    def zi_nbinom(Y, P):
        """
        Zero-Inflated Negative Binomial likelihood.
        Structural Zeroes induced by either binomial distribution, additional zeroes
        from the Negative Binomial distribution.

        Args:
            Y: observed data
            P: list with the following elements:
                0: the probability of a structural zero
                1: mean of the Poisson distribution
                2: over-dispersion parameter for negative binomial
        """
        assert P.shape[1] == 3
        p = P[0]
        theta = P[1]
        k = P[2]
        ll = (
            (np.log(p + (1 - p) * (1 + k * theta) ** (-1 / k))) * (Y == 0) +
            (np.log(1 - p) +
             loggamma(Y + k ** (-1)) -
             loggamma(k ** (-1)) -
             k ** (-1) * np.log(1 + k * theta) -
             Y * np.log(1 + (theta * k) ** (-1))) * (Y > 0)
        )
        return -ll

    @staticmethod
    def nbinom(Y, P):
        """
        Negative Binomial likelihood.

        The over-dispersion parameter P[1] follows a
        gamma(1/k, 1/k) distribution, so Var = P[1]
        Mean = 1. And a larger Var means more over-dispersion.

        Args:
            Y: observed data
            P: list with the following elements:
                0: mean of the Poisson (also negative binomial) distribution
                1: over-dispersion parameter for negative binomial
        """
        assert P.shape[0] == 2
        theta = P[0]
        k = P[1] ** -1

        ll = (
            loggamma(Y + k) - loggamma(k) +
            k * np.log(k) - k * np.log(k + theta) +
            Y * np.log(theta) - Y * np.log(theta + k)
        )

        return -ll

    @staticmethod
    def logistic(Y, P):
        """
        Logistic regression likelihood where data are 0's and 1's.
        For aggregated data, this function can still be used because
        the likelihood will be multiplied by the total number of 0's or 1's
        in each row of data.

        Args:
            Y: observed data -- should only be 1's and 0's
            P: list with the following elements:
                0: probability of the outcome Y == 1
        """
        assert P.shape[0] == 1
        p = P[0]
        assert ((Y == 1) | (Y == 0)).all()

        ll = (
                (Y == 1) * np.log(p) + (Y == 0) * np.log(1 - p)
        )

        return -ll
