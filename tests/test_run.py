import pytest
import pandas as pd
import numpy as np

from ccount.run import ModelRun
from ccount.processing import resample_data


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


def test_model_run_bootstrap_dfs(df):
    np.random.seed(10)
    df = pd.DataFrame({
        'group': np.repeat([1, 2, 3, 4, 5], repeats=2),
        'outcome': np.tile([0, 1], reps=5),
        'size': np.random.randint(low=10, high=20, size=10)
    })
    sampled_data = resample_data(
        df=df, size_col='size', outcome_col='outcome', id_cols=['group'], num_samples=10
    )
    m = ModelRun(
        model_type='logistic',
        training_df=df,
        prediction_df=df,
        outcome_variables=['outcome'],
        fixed_effects=[[[]]],
        random_effect='group',
        optimize_U=False,
        compute_D=False,
        bootstraps=10,
        weight='size',
        bootstrap_dfs=sampled_data
    )
    m.run(pools=5)
    predictions = m.predict()
    assert len(predictions) == len(df)
    assert (predictions['lower'] < predictions['mean']).all()
    assert (predictions['upper'] > predictions['mean']).all()
    assert len(predictions['lower'].unique()) == 1
    assert len(predictions['mean'].unique()) == 1
    assert len(predictions['upper'].unique()) == 1
