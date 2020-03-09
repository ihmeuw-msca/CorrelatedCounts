# Available Models

## Parametrizing a Model

Each of the available models is described below. They have different functional forms and therefore different parameters, and possibly different link functions for those parameters.

The function to run a model is `ccount.run.df_to_model`, and requires the argument `model_type`. You determine the order of the *outcomes* by the order of the elements of the list of variable names that you pass as the `outcomes` argument, but you do not determine the order of the parameters in the model (that is explained below).

The fixed effects \(\beta\) are allowed to differ across the \(n\) outcomes and \(l\) parameters. The order in which you indicate the `fixed_effects` argument matters, and that order is described below. For example, for \(n=2\) and \(l=2\),

```
fixed_effects=[
    [
        ['covariate_1'],                    # first parameter - first outcome
        ['covariate_1']                     # first parameter - second outcome
    ],
    [
        ['covariate_1', 'covariate_2'],     # second parameter - first outcome
        ['covariate_2']                     # second parameter - second outcome
    ]                    
]
```

will add `'covariate_1'` as a covariate for the first parameter on both outcomes, add `'covariate_2'` as the only covariate for the second parameter and second outcome, and add both `'covariate_1'` and `'covariate_2'` as covariates for the second parameter and first outcome.

## Model Choices

### Zero-Inflated Poisson Model

The Zero-Inflated Poisson model (ZIP) has a Poisson distribution, with a binomial distribution that determines that probability of the Poisson random variable realization being masked by an additional zero. Zeros can arise from the Poisson distribution *or* the Binomial distribution.

- **Parameter 1**: probability of a structural zero (coming from the Binomial distribution)
- **Parameter 2**: mean of the Poisson distribution

To fit this model, use `model_type = "zero_inflated_poisson"`, which will use an exponential link function for the mean and the inverse logit function for the probability of a structural zero. Alternatively, you can use `model_type = "zero_inflated_poisson_relu"`, which will use a modified link function for the mean that is more stable for large values.

### Poisson Hurdle Model

The Poisson hurdle model has a Poisson distribution that has been truncated at 0, and re-normalized for its new support, and a binomial model for the probability of a 0. In this model, zeros can only arise from the Binomial distribution.

- **Parameter 1**: probability of a zero
- **Parameter 2**: mean of the Poisson *before* truncation

To fit this model, use `model_type = "hurdle_poisson"`, which will use an exponential link function for the mean and the inverse logit function for the probability of a structural zero. Alternatively, you can use `model_type = "hurdle_poisson_relu"`, which will use a modified link function for the mean that is more stable for large values (if you haven't already scaled your covariates).

### Negative Binomial

The Negative Binomial model is an extension of the Poisson model that allows for over-dispersion (since in the Poisson distribution the mean = variance).

- **Parameter 1**: mean of the distribution
- **Parameter 2**: over-dispersion parameter (large values means variance >> mean)

To fit this model, use `model_type = "negative_binomial"`, which will use an exponential link function for the mean and the over-dispersion parameter.

### Logistic Model

The Logistic model fits a logistic regression model.

- **Parameter 1**: probability of the outcome

To fit this model, use `model_type = "logistic"`, which will use the inverse logit function for the probability of the outcome.
