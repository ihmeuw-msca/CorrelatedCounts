"""
Functions for pre-processing of data that may be helpful.
"""
from typing import List
import pandas as pd
import numpy as np


def resample_data(df: pd.DataFrame, outcome_col: str,
                  id_cols: List[str], size_col: str,
                  num_samples: int) -> List[pd.DataFrame]:
    """
    Resamples data based on observed probability.
    Need the outcome_col to be 0's and 1's.

    Parameters
    ----------
    df
    outcome_col
    id_cols
    size_col
    num_samples

    Returns
    -------
    A list of data frames that you can pass to the bootstrap function.
    """
    dfs = list()
    if not all(df[outcome_col].unique() == [0, 1]):
        raise RuntimeError(f"Outcome column needs to have 1's and 0's -- found: {df[outcome_col].unique()}.")

    sizes = df.groupby(id_cols)[size_col].sum().values
    ps = df.groupby(id_cols).apply(
        lambda row: (row[outcome_col] * row[size_col]).sum() / row[size_col].sum()
    ).values

    ids = df.sort_values(id_cols)[id_cols].drop_duplicates()

    for i in range(num_samples):

        ones = np.random.binomial(n=sizes, p=ps)
        zeroes = sizes - ones

        df_i0 = ids.copy()
        df_i1 = ids.copy()
        df_i0[outcome_col] = 0
        df_i1[outcome_col] = 1
        df_i0[size_col] = zeroes
        df_i1[size_col] = ones

        df_i = pd.concat([df_i0, df_i1], axis=0).reset_index(drop=True)
        df_i.sort_values(id_cols, inplace=True)
        dfs.append(df_i)

    return dfs
