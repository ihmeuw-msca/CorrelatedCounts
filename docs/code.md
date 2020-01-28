# ccount

# ccount.run

## initialize_model
```python
initialize_model(model_type, **kwargs)
```

Initialize a correlated count model

Args:
    model_type: (str) likelihood + parameters to fit

Returns:
    ccount.core.CorrelatedModel

## convert_df_to_model
```python
convert_df_to_model(model_type, df, outcome_variables, fixed_effects, random_effect, offset, weight, **kwargs)
```

Make a data frame with some outcomes and covariates into a
correlated model.

Args:
    model_type: (str)
    df: (pd.DataFrame)
    outcome_variables: (list)
    fixed_effects: (list)
    random_effect: (str)
    offset: (list)
    weight: (list)

Returns:
    ccount.core.CorrelatedModel

## get_predictions_from_df
```python
get_predictions_from_df(model, df, fixed_effects, random_effect, offset)
```

Add predictions to a dataset from a model that has already been fit.

Args:
    model: ccount.core.CorrelatedModel
    df: pd.DataFrame
    fixed_effects: list of list of str
    random_effect: str
    offset: list of str

Returns:
    np.array of predictions


