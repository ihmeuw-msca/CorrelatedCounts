# -*- coding: utf-8 -*-
"""
    test_utils
    ~~~~~~~~~~

    Test utils module 
"""
import numpy as np
import pytest
import ccount.utils as utils


@pytest.mark.parametrize("vec", [np.arange(10)])
@pytest.mark.parametrize("sizes",
                         [np.array([5, 5]),
                          np.array([1, 2, 3, 4])])
def test_split(vec, sizes):
    result = utils.split(vec, sizes)
    assert len(result) == len(sizes)
    s = np.cumsum(sizes)
    for i in range(len(result)):
        assert len(result[i]) == sizes[i]
        assert result[i][-1] == s[i] - 1


@pytest.mark.parametrize("vec", [np.arange(20)])
@pytest.mark.parametrize("d", [np.array([[5, 1, 2, 2], [1, 2, 3, 4]])])
def test_beta_transform(vec, d):
    beta = utils.vec_to_beta(vec, d)
    vec_recover = utils.beta_to_vec(beta)
    assert np.linalg.norm(vec - vec_recover) < 1e-10
