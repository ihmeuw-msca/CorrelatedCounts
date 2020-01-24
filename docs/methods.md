# Methods

The correlated counts framework is useful for modeling the occurrences of multiple outcomes that arise from the same individual or observational unit (e.g. location). Consider \(m\) observations, with \(n\) outcomes. For example, one might model the number of faculty (outcome n=1) and students (outcome n=2) at 100 universities. We model these in a correlated framework because it is reasonable to assume that the number of faculty at a university is likely correlated with the number of students.

Returning to the general case with \(m\) observations and \(n\) outcomes, in the linear context, we assume that the mean of the outcome is a function of the covariates for this outcome \(X_{i,j}\), the coefficients to be estimated for this outcome \(\beta{j}\), and a random effect \(U_{i,j}\)
$$
E[Y_{i,j}|X_{i,j}, \beta, U_{i,j}] = X_{i,j} \beta_{j} + U_{i,j}
$$
for the \(i^{th}\) observation and the \(j^{th}\) outcome, where \(\epsilon_{i} \sim N(0, \sigma^2)\) but with the additional assumption that the \(U_{i,}\) are multivariate normal, with mean 0 and covariance given by \(D\).
$$
U_{i,} \sim N_{n}(0, D) \quad D \in \mathbb{R}^n
$$
