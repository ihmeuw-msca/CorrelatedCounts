"""Easy functions for initializing a correlated count model."""

import numpy as np

from ccount.bsplines import spline_design_mat
from ccount.models import MODEL_DICT


def initialize_model(model_type, **kwargs):
    """
    Initialize a correlated count model

    Arguments
    ---------
        model_type: (str) likelihood + parameters to fit

    Returns
    -------
        ccount.core.CorrelatedModel
    """
    try:
        model_class = MODEL_DICT[model_type]
    except KeyError:
        raise RuntimeError(f"Cannot find model class for {model_type}. Pick one of {list(MODEL_DICT.keys())}")

    return model_class(**kwargs)


def convert_df_to_model(model_type, df, outcome_variables,
                        fixed_effects, random_effect, spline=None, offset=None, weight=None, **kwargs):
    """
    Convert a data frame to a correlated model.

    Take a data frame with some outcomes and covariates into a correlated model.

    Parameters:
        model_type: (str) the model type to run
        df: (pd.DataFrame) data frame that has all variables
        outcome_variables: (list)
        fixed_effects: (list)
        spline: (list of list of list of dict) optional
        random_effect: (str)
        offset: (list)
        weight: (list)

    Returns:
        ccount.core.CorrelatedModel
    """
    assert type(model_type) == str
    assert type(outcome_variables) == list
    for f in outcome_variables:
        assert type(f) == str

    assert type(fixed_effects) == list
    for f in fixed_effects:
        assert type(f) == list, "fixed_effects must be a list of lists"
        for g in f:
            assert (type(g) == list) or (g is None)
            if g is not None:
                for c in g:
                    assert c in df.columns
                    df = df.loc[~df[c].isnull()]

    if spline is not None:
        for s in spline:
            assert type(s) == list
            for g in s:
                assert (type(g) == list) or (g is None)
                if g is not None:
                    for c in g:
                        assert c['name'] in df.columns
                        df = df.loc[~df[c['name']].isnull()]

    assert type(random_effect) == str
    assert random_effect in df.columns

    if offset is not None:
        assert type(offset) == list
        for o in offset:
            assert (type(o) == str) or (o is None)
            if o is not None:
                assert o in df.columns

    X = [
        [np.asarray(df[g]) if g is not None else None for g in f]
        for f in fixed_effects
    ]
    if spline is not None:
        for s in spline:
            for g_dict in s:
                if g_dict is not None:
                    for g in g_dict:
                        g.update({'spline_var': np.asarray(df[g['name']])})

    Y = np.asarray(df[outcome_variables])

    # Get random effects, offsets
    group_id = np.asarray(df[[random_effect]]).astype(int).ravel()
    if offset is not None:
        offsets = [np.asarray(df[[o]]) if o is not None else None for o in offset]
    else:
        offsets = offset

    if weight is not None:
        weight = np.asarray(df[[weight for i in range(Y.shape[1])]])
    d = np.array([[x.shape[1] if x is not None else 0 for x in k] for k in X])

    return initialize_model(
        model_type=model_type,
        m=Y.shape[0],
        n=Y.shape[1],
        d=d,
        Y=Y, X=X, spline_specs=spline, group_id=group_id,
        offset=offsets,
        weights=weight,
        **kwargs
    )


def get_predictions_from_df(model, df,
                            fixed_effects, random_effect, spline=None, offset=None):
    """
    Add predictions to a dataset from a model that has already been fit.

    Args:
        model: ccount.core.CorrelatedModel
        df: pd.DataFrame
        fixed_effects: list of list of list of str
        random_effect: str
        spline: list of list of str
        offset: list of str

    Returns:
        np.array of predictions

    """
    for f in fixed_effects:
        for g in f:
            if g is not None:
                for c in g:
                    df = df.loc[~df[c].isnull()]
    X = [
        [np.asarray(df[g]) if g is not None else None for g in f]
        for f in fixed_effects
    ]
    if spline is not None:
        for s in spline:
            for g_dict in s:
                if g_dict is not None:
                    for g in g_dict:
                        g.update({'spline_var': np.asarray(df[g['name']])})

    if offset is not None:
        offsets = [np.asarray(df[[o]]) if o is not None else None for o in offset]
    else:
        offsets = None
    group_id = np.asarray(df[[random_effect]]).astype(int).ravel()
    return np.transpose(
        model.predict(
            X=X, m=len(df),
            spline_specs=spline,
            group_id=group_id,
            offset=offsets
        )
    )
