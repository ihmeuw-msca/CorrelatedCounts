# Methods

### Statistical Model

The correlated counts framework is useful for modeling the occurrences of multiple outcomes that arise from the same individual or observational unit (e.g. location). Consider \(m\) observations, with \(n\) outcomes. For example, one might model the number of faculty (outcome n=1) and students (outcome n=2) at 100 universities. We model these in a correlated framework because it is reasonable to assume that the number of faculty at a university is likely correlated with the number of students.

Returning to the general case with \(m\) observations and \(n\) outcomes, in the linear context, we assume that the mean of the outcome is a function of the covariates for this outcome \(X_{i,j}\), the coefficients to be estimated for this outcome \(\beta_{j}\), and a random effect \(U_{i,j}\)
$$
E[Y_{i,j}|X_{i,j}, \beta, U_{i,j}] = X_{i,j} \beta_{j} + U_{i,j}
$$
for the \(i^{th}\) observation and the \(j^{th}\) outcome, where \(\epsilon_{i} \sim N(0, \sigma^2)\) but with the additional assumption that the \(U_{i,}\) are multivariate normal, with mean 0 and covariance given by \(D\).
$$
U_{i,.} \sim N_{n}(0, D) \quad D \in \mathbb{R}^n
$$
The \(U_{i,j}\)'s are correlated with one another, and that correlation drives correlation in the mean. This method follows that outlined by [Rodrigues-Motta and colleages (2012)](https://www.tandfonline.com/doi/full/10.1080/02664763.2013.789098?scroll=top&needAccess=true).

In this simple case, \(Y\) follows a normal distribution, but when working with counts, it is more common to use discrete distributions like the Poisson distribution or the Negative Binomial distribution. In cases where we have extremely rare events, we may also consider extensions to these distributions that allow for more zeros than would be typically realized in the discrete distributions (e.g. zero-inflation, hurdle models). As such, we will usually have more than one parameter to estimate besides the *mean* outcome.

Most generally, consider that now we have \(l\) parameters. We have some probability distribution for \(Y\), that is dependent on \(X_{i,j,k}\), \(\beta_{j,k}\) and \(U_{i,j,k}\),
$$
f(Y_{i,j}|X_{i,j,k}, \beta_{j,k}, U_{i,j,k}) \quad k = 1, ..., l
$$
where the covariates \(X\) and random effects \(U\) can differ with respect to each of the \(l\) parameters of the discrete distribution (e.g. the mean and the over-dispersion parameter for the variance in the Negative Binomial distribution). Depending on the support for the parameter, we may need to transform the linear combination of \(X_{i,j,k} \beta_{j,k} + U_{i,j,k}\) into the space that makes sense for the parameter and distribution at hand. For example, the mean of the Poisson distribution must be \(>0\), so a natural *link function* that does this transformation is \(e^{X_{i,j,k} \beta_{j,k} + U_{i,j,k}}\).

### Splines

Depending on the data generating process, you may want to use a more flexible functional form for modeling the mean over a variable rather than 
Examples would include age or time variables. We recommend only putting splines on the mean parameter.

To model the relationship in the mean function over a specific variable, we use B-splines. You may use as many
B-splines as you want, but keep in mind that they are all 1-dimensional splines per variable rather than an N-dimensional spline surface over the combinations of your variables.

### Offsets

Count data frequently arises from populations of varying sizes. For example, a count of deaths of 10 in a population of 100 represents a higher death rate than a 10 deaths in a population of 10,000. Therefore, when you have a model that you have parametrized such that some parameter affects the *mean* number of deaths, you will want to offset that mean by the population size.

Adding an offset for the mean model means that you are effectively modeling the rate per offset unit, rather than the count alone. *It is not advised to add an offset to parameters other than the mean.*[^1]

[^1]: For the Poisson, Zero-Inflated Poisson, and Negative Binomial models, the offset on the mean translates to directly modeling the rate (with more weight given to data points with larger denominators), but for the Hurdle models, the offset on the mean is only an approximation of the rate.

### Likelihood Weights

You may want to weight data points differently depending on how certain those data points are relative to others. This is implemented with a weighted likelihood approach: the likelihood contribution for each data point is multiplied by its corresponding weight (if weights are supplied). This will up-weight the contribution of more certain data points to the estimation problem, and down-weight the contribution of more uncertain data points.

### Optimization

To estimate the parameters of this model, we add a prior for \(U\) that incorporates \(D\), and then optimize \(f(Y_{i,j}|...)\) with respect to \(\beta\), \(U\) and \(D\).
