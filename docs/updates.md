# Release Notes

## March XX, 2020 (v0.0.2)
- *Feature*: Added a new functionality to run models (see [here](code.md#easy-model-launching))
- *Feature*: If you use the new functionality above, you can do a data bootstrap to produce uncertainty

## March 9, 2020 (v0.0.1)
- *Feature*: Added a logistic regression model, see [model specification](models.md#logistic-model)
- *Bugfix*: Models with only one outcome (when computing variance-covariance matrix,
 have to adjust the shape of the output since there no covariance with only one outcome)
- *Bugfix*: For weights when only specifying one outcome (previous weights assumed two outcomes)
- *Bugfix*: Weights were not being sorted in the same way as the other data inputs

## February 18, 2020 (v0.0.0)
- *Feature*: Include splines for any parameter and outcome, see [model specification](code.md#specification)
- *Feature*: Added optimization iteration flexibility for fixed and random effects, plus optional relative tolerance, see [optimization](code.md#optimization)
- *Bugfix*: the `ccount.core.CorrelatedModel.summarize()` function
