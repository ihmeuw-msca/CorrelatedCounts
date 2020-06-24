# Running a Model

The core model is an object called `ccount.core.CorrelatedModel`. Each of the types of models you can fit are explained in the [models](models.md) documentation.
If you are new to using this code base, please read the following sections about [model specification](#specifying-the-model),
[fitting the model](#fitting-the-model), and [creating predictions](#creating-predictions).

We have recently added a `ModelRun` class that combines all of the steps in the aforementioned sections to limit the amount
of code that you need to write. It also allows you to use the data bootstrap method to produce confidence intervals for your predictions. See [here](#easy-model-launching) for documentation.

## Specifying the Model

To specify a model, make sure that all variables that you need are contained within one data frame, including covariates, random effects, offsets, and weights. The outcome variables should be wide, but all other variables should be long, e.g. have one column for deaths and another column for cases, rather than one column with deaths and cases stacked on top of each other. All of these variables need to be filled in for every entry (i.e. there can be no missing values -- if you want to predict for somewhere with missing outcome information, you can do that later with the [function](#creating-predictions) that makes predictions).

The class that you can use to initialize a model is `ccount.run.ModelRun`.
This class makes it easy to fit the model and to predict for a new data frame. It takes the following arguments:

- `model_type`: `(str)` The name of the model type to be fit. The list of names are in [models](models.md).
- `df`: `(pd.DataFrame)` A data frame that contains all of your variables.
- `outcome_variables`: `(List[str])` A list of the names of the outcome variables.
- `fixed_effects`: `(List[List[List[str] or None]])` Nested lists of the names of fixed effects to put on each of the parameters and outcomes. See [here](models.md#parametrizing-a-model) for details. If you do **not** want to put any covariates on a parameter-outcome combination, pass `None` instead of `List[str]`. *Note: Intercepts are automatically added for each parameter-outcome. Do not add another intercept.*
- `random_effect`: `(str)` Name of the variable that specifies the grouping of the random effect.
- `spline`: `(List[List[List[dict] or None]])` Nested lists of dictionaries that gives the spline specification
- `offset`: `List[str]` (optional) List of variable names to use as an offset for each parameter. Must be of the length of the number of parameters in each model, **and in the correct order**. See [here](models.md#model-choices) for the number and order of parameters for each model type. To include an offset on only one parameter, pass a list of the variable name and `None`, in the correct order corresponding to your model type.
- `weight`: `(str)` (optional) Name of the variable that specifies the weight to place on each observation.
- `**kwargs`: Additional arguments
    + `normalize_X`: `(bool)` Whether or not to scale the covariates by their mean and standard deviation. By default, `normalize_X = True`. The resulting parameters are transformed after fitting so that they can be interpreted in the original space as the covariates.
    + `add_intercepts`: `(bool)` Whether or not to add intercepts for all parameter-outcomes. By default, `add_intercepts = True`.

#### Spline Specification

Fitting a spline requires additional information. For each spline that you want to add, you pass a dictionary rather than a string (like you do for fixed effects).
The dictionary must have the following keys:

- `name`: `(str)` variable name in your dataset for the spline
- `knots_type`: `(str)` one of `"frequency"` (put the knots at equal quantiles of the data) or `"domain"` (put the knots at equal spacings over the domain of the variable)
- `knots_num`: `(int)` number of knots for the spline
- `degree`: `(int)` degree of differentiation for the spline (e.g. degree of 3 is a cubic spline, most common for this application)
- `r_linear`: `(bool)` enforce linear function (rather than a potentially degree > 1 spline) at the tails of your data on the *right*
- `l_linear`: `(bool)` enforce linear function (rather than a potentially degree > 1 spline) at the tails of your data on the *left*

### Example Specification

Consider a pandas data frame, `df`, that has the following columns: `deaths`, `cases`, `median_income`, `population`, `country`, `age`. In my example below, I will fit a Zero-Inflated Poisson model (described [here](models.md#zero-inflated-poisson-model)) with the following:

*Outcomes*

- predict both deaths due to and cases of a disease, where deaths and cases are correlated with one another (2-dimensional outcome)

*Fixed Effects*

- median income as a predictor for the mean of deaths
- age as a predictor for the mean of cases
- no predictors for the probability of a structural zero

*Splines*
- a spline over calendar year with 3 degrees of freedom (cubic), and 3 knots (two at the min and max of the variable, one at the median), no linearity in tails
- only applied to the mean, but applied to both cases and deaths

Let's specify the spline here, and in the example we will apply the spline specification to the mean function for both outcomes.

```
spline_specs = {
    'name': 'year',
    'knots_type': 'frequency',
    'knots_num': 3,
    'degree': 3,
    'r_linear': False,
    'l_linear': False
}
```

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
    offset=[None, 'population'],
    spline=[[None, None],
        [[spline_specs], [spline_specs]]
    ]
)
```

## Fitting the Model

Once you have the model object, returned by the function `model = convert_df_to_model(...)`, we can estimate the parameters. The class method `ccount.core.CorrelatedModel.optimize_params` does the optimization work.

From our example above, to fit the model, we simply run:

```
model.optimize_params(max_iters=100, max_beta_iters=10, max_U_iters=10, rel_tol=1e-4)
```

The parameters that are being optimized are the fixed effects and random effects. The optimization happens iteratively,
and you may not need to completely optimize the fixed and random effects in each overall iteration. To control this process, you 
may pass in any integer for `max_iters`, `max_beta_iters`, and `max_U_iters`. You can also
include a relative tolerance for the total error in the fixed and random effects with the argument `rel_tol`, and the 
program will terminate if it reaches `rel_tol` before completing `max_iters` iterations. We recommend the defaults above for all
of these arguments.

The parameter estimates, including the \(\beta\) fixed effects, the \(U\) random effects, and the correlation between the outcomes given by \(D\) (each described in [methods](methods.md)) are all available in the `summarize` class method for `ccount.core.CorrelatedModel`. In our example above, to get a printed summary of the estimates (both transformed and un-transformed based on the link functions described in [the model choices](models.md#model-choices)), run the following:

```
model.summarize(file=...)
```

If you want to save the model summary to a file so that you can access it later, pass `file=...` and it will re-route the output to whatever file name (must have a `.txt` extension) that you pass.
If you do not pass a file, i.e. `file=None`, then it will return the summary in your session.

## Creating Predictions

In order to get the fitted values of deaths and cases, you can use the function `ccount.run.get_predictions_from_df`. It takes the same arguments as `convert_df_to_model`, except that in place of `model_type`, it needs a `ccount.core.CorrelatedModel` object, and it does not need the outcome variables. For our example above, this looks like

`predictions` is a 2-dimensional array that is the length of the data frame `some_df` along one axis and the outcome predictions along the other (i.e. predictions for deaths and cases).

The `fixed_effects`, `random_effect`, `offset` (if applicable), and `weight` (if applicable) arguments must be identical to those used in the initial model [specification](#specifying-the-model).

*Note that the data frame passed to `get_predictions_from_df` need not be the data frame that was used in model fitting.* It can be a new data frame with missing outcome variables because they are not used in making predictions since the model has already been fit. However, this new data frame cannot have any missing values for random effects, fixed effects, or offsets. If it includes random effect grouping levels that were not observed in the fitting of the model, the random effect will be 0 for that level.

## Easy Model Launching

To run a model with less code, you can use the following class that takes all of the same arguments
as `convert_df_to_model` (see [here](#specifying-the-model)), `optimize_params` (see [here](#fitting-the-model))
with some exceptions:
```
from ccount.run import ModelRun

model = ModelRun(training_df, prediction_df, bootstraps=None, **kwargs)
```
- `training_df`: `(pd.DataFrame)` your dataset that you want to train the model on (previously called `df` in `convert_df_to_model`)
- `prediction_df`: `(pd.DataFrame)` your dataset that you want to make predictions for (previously called `df` in `get_predictions_from_df`)
- **NEW**: `bootstraps`: `(optional int)` number of data bootstraps to perform for uncertainty
- `**kwargs`: all arguments that were previously passed to `convert_df_to_model` and `optimize_params`

When you initialize the `ModelRun`, it will set up your main model and then each of the models that will
be used for the data bootstraps (see [here](methods.md#data-bootstrap) for a description of the data bootstrap method).
To fit the model, use the function `ModelRun.run()`. If you are using the boostrap functionality,
you can fit the model for each bootstrapped dataset in parallel with multiprocessing pools
 by passing a number of pools to `run`, like

```
model.run(pools=5)
```

To make predictions, use the function `ModelRun.predict(alpha=0.05)`, where
`alpha` corresponds to a `1 - alpha` confidence level (e.g. 95% confidence interval). There is no need to pass
additional arguments to the `run()` and `predict()` functions because all information about the optimization and 
prediction data frame was passed in to the `ModelRun` `**kwargs**`.

