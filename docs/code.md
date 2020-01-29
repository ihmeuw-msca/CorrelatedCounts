# [Running a Model](#running)

The core model is an object called `ccount.core.CorrelatedModel`. Each of the types of models you can fit are explained in the [models](models.md) documetation.

## [Specifying the Model](#specification)

To specify a model, make sure that all variables that you need are contained within one data frame, including covariates, random effects, offsets, and weights. The outcome variables should be wide, but all other variables should be long, e.g. have one column for deaths and another column for cases, rather than one column with deaths and cases stacked on top of each other. All of these variables need to be filled in for every entry (i.e. there can be no missing values -- if you want to predict for somewhere with missing outcome information, you can do that later with the [function](code.md#predictions) that makes predictions).

The function that you can use to initialize a model is `ccount.run.convert_df_to_model`. This function returns a `CorrelatedModel` object. It takes the following arguments:

- `model_type`: `(str)` The name of the model type to be fit. The list of names are in [models](models.md).
- `df`: `(pd.DataFrame)` A data frame that contains all of your variables.
- `outcome_variables`: `(List[str])` A list of the names of the outcome variables.
- `fixed_effects`: `(List[List[List[str] or None]])` Nested lists of the names of fixed effects to put on each of the parameters and outcomes. See [here](models.md#parametrization) for details. If you do **not** want to put any covariates on a parameter-outcome combination, pass `None` instead of `List[str]`. *Note: Intercepts are automatically added for each parameter-outcome. Do not add another intercept.*
- `random_effect`: `(str)` Name of the variable that specifies the grouping of the random effect.
- `offset`: `List[str]` (optional) List of variable names to use as an offset for each parameter. Must be of the length of the number of parameters in each model, **and in the correct order**. See [here](models.md#choices) for the number and order of parameters for each model type. To include an offset on only one parameter, pass a list of the variable name and `None`, in the correct order corresponding to your model type.
- `weight`: `(str)` (optional) Name of the variable that specifies the weight to place on each observation.
- `**kwargs`: Additional arguments
    + `normalize_X`: `(bool)` Whether or not to scale the covariates by their mean and standard deviation. By default, `normalize_X = True`. The resulting parameters are transformed after fitting so that they can be interpreted in the original space as the covariates.
    + `add_intercepts`: `(bool)` Whether or not to add intercepts for all parameter-outcomes. By default, `add_intercepts = True`.

### Example Specification

Consider a pandas data frame, `df`, that has the following columns: `deaths`, `cases`, `median_income`, `population`, `country`, `age`. In my example below, I will fit a Zero-Inflated Poisson model (described [here](models.md#zip)) with the following:

*Outcomes*

- predict both deaths due to and cases of a disease, where deaths and cases are correlated with one another (2-dimensional outcome)

*Fixed Effects*

- median income as a predictor for the mean of deaths
- age as a predictor for the mean of cases
- no predictors for the probability of a structural zero

*Random Effect*

- random effect grouping by country, so that all data points within the same country will have the same 2-dimensional realization that correlates their deaths with their cases

*Offset*

- population as the "offset" for the mean, or the denominator that deaths and cases came from

```
from ccount.run import convert_df_to_model

model = convert_df_to_model(
    model_type='zero_inflated_poisson',
    df=df,
    outcome_variables=['deaths', 'cases'],
    fixed_effects=[[None, None], [['median_income'], ['age']]],
    random_effect='country',
    offset=[None, 'population']
)
```

## [Fitting the Model](#optimization)

Once you have the model object, returned by the function `model = convert_df_to_model(...)`, we can estimate the parameters. The class method `ccount.core.CorrelatedModel.optimize_params` does the optimization work.

From our example above, to fit the model, we simply run:

```
model.optimize_params(max_iters=10)
```

You may pass in any integer for `max_iters`. A sensible range for the optimization routine implemented in this package is between 5-10 iterations.

The parameter estimates, including the \(\beta\) fixed effects, the \(U\) random effects, and the correlation between the outcomes given by \(D\) (each described in [methods](methods.md)) are all available in the `summarize` class method for `ccount.core.CorrelatedModel`. In our example above, to get a printed summary of the estimates (both transformed and un-transformed based on the link functions described in [the model choices](models.md#choices)), run the following:

```
model.summarize()
```

If you want to save the model summary to a file so that you can access it later, pass `file=...` and it will re-route the output to whatever file name (must have a `.txt` extension) that you pass.

## [Creating Predictions](#predictions)

In order to get the fitted values of deaths and cases, you can use the function `ccount.run.get_predictions_from_df`. It takes the same arguments as `convert_df_to_model`, except that in place of `model_type`, it needs a `ccount.core.CorrelatedModel` object, and it does not need the outcome variables. For our example above, this looks like

```
from ccount.run import get_predictions_from_df

predictions = get_predictions_from_df(
    model=model,
    df=some_df,
    fixed_effects=[[None, None], [['median_income'], ['age']]],
    random_effect='country',
    offset=[None, 'population']
)
```

`predictions` is a 2-dimensional array that is the length of the data frame `some_df` along one axis and the outcome predictions along the other (i.e. predictions for deaths and cases).

The `fixed_effects`, `random_effect`, `offset` (if applicable), and `weight` (if applicable) arguments must be identical to those used in the initial model [specification](code.md#specification).

*Note that the data frame passed to `get_predictions_from_df` need not be the data frame that was used in model fitting.* It can be a new data frame with missing outcome variables because they are not used in making predictions since the model has already been fit. However, this new data frame cannot have any missing values for random effects, fixed effects, or offsets. If it includes random effect grouping levels that were not observed in the fitting of the model, the random effect will be 0 for that level.