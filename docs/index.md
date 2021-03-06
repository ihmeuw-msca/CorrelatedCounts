# Welcome to CorrelatedCounts

## Getting Started

In order to run the correlated count models, you will need to install Python to your computer. It is most helpful to use [miniconda](https://docs.conda.io/en/latest/miniconda.html), and then create an environment that will be used specifically with the correlated count package. The dependencies for this package are `numpy` and `scipy`.

## Installation

To install the package, clone the GitHub repository here:
```
git clone https://github.com/mbannick/CorrelatedCounts.git
```

Once you have cloned the package, activate your conda envirionment and run
```
cd CorrelatedCounts
python setup.py install
```

The package name is `ccount`.

## Recommended Usage

We recommend using Jupyter Notebooks (documentation [here](https://jupyter-notebook.readthedocs.io/en/stable/)) in your environment, which you can install with

```
pip install jupyter
```

and then running the command

```
jupyter-notebook
```

Once inside the notebook, you can import all of the functions that are described in the [code documentation](code.md#running), including model [specification](code.md#specification), [fitting](code.md#optimization), and [prediction](code.md#predictions).