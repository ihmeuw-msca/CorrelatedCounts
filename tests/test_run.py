import pytest
import pandas as pd
import numpy as np
from time import time

from ccount.run import ModelRun


@pytest.fixture
def df():
    x1 = np.random.randn(100)
    x2 = np.random.randn(100)
    p = np.exp(x1 + x2) / (1 + np.exp(x1 + x2))
    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'group': np.repeat([0], repeats=100),
        'y': np.random.binomial(n=1, p=p)
    })


def test_model_run(df):
    m = ModelRun(
        model_type='logistic',
        training_df=df,
        prediction_df=df,
        outcome_variables=['y'],
        fixed_effects=[[['x1', 'x2']]],
        random_effect='group',
        optimize_U=False,
        compute_D=False
    )
    m.run()
    predictions = m.predict()
    assert len(predictions) == len(df)


def test_model_run_bootstrap(df):
    np.random.seed(10)
    m = ModelRun(
        model_type='logistic',
        training_df=df,
        prediction_df=df,
        outcome_variables=['y'],
        fixed_effects=[[['x1', 'x2']]],
        random_effect='group',
        optimize_U=False,
        compute_D=False,
        bootstraps=10
    )
    m.run()
    predictions = m.predict()
    assert len(predictions) == len(df)
    assert (predictions['lower'] < predictions['mean']).all()
    assert (predictions['upper'] > predictions['mean']).all()


def test_model_run_bootstrap_pools(df):
    np.random.seed(10)
    m = ModelRun(
        model_type='logistic',
        training_df=df,
        prediction_df=df,
        outcome_variables=['y'],
        fixed_effects=[[['x1', 'x2']]],
        random_effect='group',
        optimize_U=False,
        compute_D=False,
        bootstraps=10
    )
    m.run(pools=5)
    predictions = m.predict()
    assert len(predictions) == len(df)
    assert (predictions['lower'] < predictions['mean']).all()
    assert (predictions['upper'] > predictions['mean']).all()
