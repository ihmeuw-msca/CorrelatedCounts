import numpy as np

from ccount.models import MODEL_DICT


def initialize_model(model_type, **kwargs):
    """
    Initialize a correlated count model

    Args:
        model_type: (str) likelihood + parameters to fit

    Returns:
        ccount.core.CorrelatedModel
    """
    try:
        model_class = MODEL_DICT[model_type]
    except KeyError:
        raise RuntimeError(f"Cannot find model class for {model_type}. Pick one of {list(MODEL_DICT.keys())}")

    return model_class(**kwargs)


def df_to_model(model_type, df, outcome_variables,
                fixed_effects, random_effect, offset, **kwargs):
    """
    Make a data frame with some outcomes and covariates into a
    correlated model.

    Args:
        model_type: (str)
        df: (pd.DataFrame)
        outcome_variables: (list)
        fixed_effects: (list)
        random_effect: (str)
        offset: (str)

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

    assert type(random_effect) == str
    assert random_effect in df.columns

    assert type(offset) == list
    for o in offset:
        assert (type(o) == str) or (o is None)
        if o is not None:
            assert o in df.columns

    X = [
        [np.asarray(df[g]) if g is not None else None for g in f]
        for f in fixed_effects
    ]
    Y = np.asarray(df[outcome_variables])

    # Get random effects, offsets
    group_id = np.asarray(df[[random_effect]]).astype(int).ravel()
    offsets = [np.asarray(df[[o]]) if o is not None else None for o in offset]
    d = np.array([[x.shape[1] if x is not None else 0 for x in k] for k in X])

    return initialize_model(
        model_type=model_type,
        m=Y.shape[0],
        n=Y.shape[1],
        d=d,
        Y=Y, X=X, group_id=group_id,
        offset=offsets,
        **kwargs
    )


def add_predictions_to_df(model, df,
                          fixed_effects, random_effect, offset):
    """
    Add predictions to a dataset from a model that has already been fit.

    Args:
        model: ccount.core.CorrelatedModel
        df: pd.DataFrame
        fixed_effects: list of list of str
        random_effect: str
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
    offsets = [np.asarray(df[[o]]) if o is not None else None for o in offset]
    group_id = np.asarray(df[[random_effect]]).astype(int).ravel()
    return np.transpose(
        model.predict(
            X=X, m=len(df),
            group_id=group_id,
            offset=offsets
        )
    )