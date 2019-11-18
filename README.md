[![Build Status](https://travis-ci.com/mbannick/CorrelatedCounts.svg?branch=master)](https://travis-ci.com/mbannick/CorrelatedCounts)

# CorrelatedCounts
Modeling correlated count data based off of the model in Rodrigues-Motta et al 2013.<sup>1</sup>

## Installation
```
git clone https://github.com/mbannick/CorrelatedCounts.git
cd CorrelatedCounts
python setup.py install
```

## Usage
### Simulate corrlated counts with zero-inflated Poisson model

Zeroes can appear through two mechanisms: (1) structural zeros with probability p, and (2) zeroes from the Poisson
distribution. The baseline probability of a structural zero is 0.5.

```ipython
In [1]: from ccount.simulate import PoissonSimulation

In [2]: s = PoissonSimulation(n=3, J=2, d=[2, 2])

In [3]: s.simulate() # baseline probability of observing a structural zero = 0.5
Out[3]:
array([[0, 0, 0],
       [1, 0, 2]])

In [4]: s.simulate()
Out[4]:
array([[ 0,  1,  1],
       [22,  0,  0]])

In [5]: s.update_params(p=0.9) # update the probability of a structural zero to 0.9

In [6]: s.simulate()
Out[6]:
array([[0, 0, 0],
       [2, 0, 0]])

In [7]: s.simulate()
Out[7]:
array([[0, 0, 0],
       [0, 0, 0]])
```

### References

<sup>1</sup> Mariana Rodrigues-Motta, Hildete P. Pinheiro, Eduardo G. Martins, Márcio S. Araújo & Sérgio F. dos Reis (2013) Multivariate models for correlated count data, Journal of Applied Statistics, 40:7, 1586-1596, DOI: 10.1080/02664763.2013.789098

