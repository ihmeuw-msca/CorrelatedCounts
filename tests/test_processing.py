import pytest
import numpy as np
import pandas as pd

from ccount.processing import resample_data


@pytest.fixture
def df():
    return pd.DataFrame({
        'group': np.repeat([1, 2, 3, 4, 5], repeats=2),
        'outcome': np.tile([0, 1], reps=5),
        'size': np.random.randint(low=10, high=20, size=10)
    })


def test_resample_data(df):
    np.random.seed(10)
    dfs = resample_data(
        df=df, outcome_col='outcome',
        size_col='size', id_cols=['group'], num_samples=10
    )
    assert len(dfs) == 10
    for i in dfs:
        assert (i.columns == df.columns).all()
        assert len(i) == len(df)
        np.testing.assert_array_equal(
            i['group'].unique(),
            df['group'].unique()
        )
        np.testing.assert_array_equal(
            i.groupby(['group'])['size'].sum().values,
            df.groupby(['group'])['size'].sum().values
        )
