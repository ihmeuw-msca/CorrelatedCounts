# -*- coding: utf-8 -*-
"""
    Zero-Inflated Poisson Demo
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from ccount.simulate import PoissonSimulation
from ccount.models import ZeroInflatedPoisson
import numpy as np


# set up the simulation
m = 100
n = 2
d = [1]*n

p = 0.1
x = [np.ones((m, d[j])) for j in range(n)]
beta = [np.array([0.1]*d[j]) for j in range(n)]
D = np.array([[1.0, 0.1], [0.1, 1.0]])

s_zip = PoissonSimulation(m=m, n=n, d=d)
s_zip.update_params(p=p, x=x, beta=beta, D=D)

Y = s_zip.simulate()

# fit the model
d = np.array([[1]*n, d])
X = [[np.ones((m, 1)) for j in range(n)], x]
m_zip = ZeroInflatedPoisson(m=m, d=d, Y=Y, X=X)
m_zip.optimize_params()

