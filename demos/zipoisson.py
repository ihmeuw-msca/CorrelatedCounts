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

p = 0.3
x = [np.ones((m, d[j])) for j in range(n)]
beta = [np.array([0.1]*d[j]) for j in range(n)]
D = np.array([[1.0, 0.1], [0.1, 1.0]])

s_zip = PoissonSimulation(m=m, n=n, d=d)
s_zip.update_params(p=p, x=x, beta=beta, D=D)

Y = s_zip.simulate()

# fit the model
d = np.array([[1]*n, d])
X = [[np.ones((m, 1)) for j in range(n)], x]

beta = [[np.array([np.log(p/(1.0 - p))]), np.array([np.log(p/(1.0 - p))])],
        [np.array([0.1]), np.array([0.1])]]
D = np.array([np.eye(n)*1e-10, D])
U = np.array([np.zeros(s_zip.u.shape), s_zip.u])
theta = np.vstack(s_zip.theta).T
P = np.array([np.ones(theta.shape)*p, theta])

m_zip = ZeroInflatedPoisson(m=m, d=d, Y=Y, X=X)
m_zip.update_params(D=D, U=U)
m_zip.optimize_params(optimize_U=False, compute_D=False, max_iters=1)

