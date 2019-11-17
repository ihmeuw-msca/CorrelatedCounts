import pytest
import numpy as np

from ccount.simulate import Simulation, PoissonSimulation


@pytest.fixture
def m():
    return 10


@pytest.fixture
def n():
    return 2


@pytest.fixture
def d():
    return [2, 2]


@pytest.fixture
def poisson_sim():
    return np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [194, 0, 0, 3, 1, 2, 0, 2, 0, 0]]
    )


def test_simulate(m, n, d):
    s = Simulation(m=m, n=n, d=d)
    assert s.m == 10
    assert s.n == 2
    assert s.d == [2, 2]


def test_poisson_simulation(m, n, d, poisson_sim):
    np.random.seed(0)

    s = PoissonSimulation(m=m, n=n, d=d)
    assert s.p == 0.5
    s.update_params(p=0.4)
    assert s.p == 0.4
    assert s.D.shape == (2, 2)
    assert all([i.shape == (m, n) for i in s.x])
    assert all([i.shape == (2,) for i in s.beta])

    sim = s.simulate()
    assert (sim == poisson_sim).all()

    assert s.u.shape == (m, n)
    assert all([i.shape == (m,) for i in s.theta])
