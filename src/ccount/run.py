"""Easy functions for initializing a correlated count model."""

import numpy as np
import logging
import pandas as pd

from ccount.models import MODEL_DICT

LOG = logging.getLogger(__name__)


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


class ModelRun:
    def __init__(self, model_type, training_df, prediction_df,
                 outcome_variables, fixed_effects, random_effect,
                 spline=None, offset=None, weight=None,
                 max_iters=100, max_beta_iters=10, max_U_iters=10, rel_tol=None,
                 optimize_beta=True, optimize_U=True, compute_D=True,
                 bootstraps=None):

        self.model_type = model_type
        self.training_df = training_df
        self.prediction_df = prediction_df

        self.outcome_variables = outcome_variables
        self.fixed_effects = fixed_effects
        self.random_effect = random_effect
        self.spline = spline
        self.offset = offset
        self.weight = weight

        self.max_iters = max_iters
        self.max_beta_iters = max_beta_iters
        self.max_U_iters = max_U_iters
        self.rel_tol = rel_tol
        self.optimize_beta = optimize_beta
        self.optimize_U = optimize_U
        self.compute_D = compute_D

        self.bootstraps = bootstraps

        self.model = None
        self.models = list()

        self.initialize()

    def initialize(self):
        """
        Initializes the mean model for all of the training data.
        """
        LOG.info("Initializing Model.")
        self.model = self.convert(df=self.training_df)
        if self.bootstraps is not None:
            LOG.info("Bootstrapping Data.")
            self.bootstrap_data()

    def run(self):
        """
        Runs the model(s).
        """
        LOG.info("Optimizing main model.")
        self.optimize(model=self.model)
        for i, mod in enumerate(self.models):
            LOG.info(f"Optimizing bootstrap model {i}.")
            self.optimize(model=self.models[i])

    def predict(self, alpha=0.05):
        """
        Creates predictions for the prediction data frame.
        Args:
            alpha: (float) confidence level for the confidence interval
        Returns:
            (pd.DataFrame)
        """
        assert 0 < alpha < 1
        predictions = self.predictions(model=self.model)
        draws = np.vstack([
            self.predictions(model=mod) for mod in self.models
        ])
        return pd.DataFrame({
            'mean': predictions[0],
            'lower': np.quantile(draws, q=alpha/2, axis=0),
            'upper': np.quantile(draws, q=1-alpha/2, axis=0)
        })

    def convert(self, df):
        """
        Helper function to convert a data frame to a model based on the attributes
        of this ModelRun.

        Args:
            df: (pandas.DataFrame)

        Returns:
            ccount.core.CorrelatedModel
        """
        return convert_df_to_model(
            model_type=self.model_type, df=df,
            outcome_variables=self.outcome_variables,
            fixed_effects=self.fixed_effects,
            random_effect=self.random_effect,
            spline=self.spline,
            offset=self.offset,
            weight=self.weight
        )

    def optimize(self, model):
        """
        Helper function to optimize a model based on the attributes of this ModelRun.
        Args:
            model: (ccount.core.CorrelatedModel)

        Returns:
            None
        """
        model.optimize_params(
            max_iters=self.max_iters, max_beta_iters=self.max_beta_iters,
            max_U_iters=self.max_U_iters, rel_tol=self.rel_tol,
            optimize_beta=self.optimize_beta, optimize_U=self.optimize_U,
            compute_D=self.compute_D
        )

    def predictions(self, model):
        """
        Helper function to get predictions from a model based on the attributes of this ModelRun
        Args:
            model: (ccount.core.CorrelatedModel)

        Returns:
            np.array
        """
        return get_predictions_from_df(
            model=model, df=self.prediction_df,
            fixed_effects=self.fixed_effects,
            random_effect=self.random_effect,
            spline=self.spline,
            offset=self.offset,
        )

    def bootstrap_data(self):
        """
        Bootstraps the data with self.bootstraps samples.
        """
        for i in range(self.bootstraps):
            df_i = self.training_df.groupby(
                self.random_effect, group_keys=False
            ).apply(
                lambda x: x.sample(len(x), replace=True)
            )
            self.models.append(self.convert(df=df_i))








