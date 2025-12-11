# Causal Estimators

This skill covers the functions and methods used to estimate and summarize causal effects after fitting a model.

## Core Estimator Methods

Most experiment classes in CausalPy (like `DifferenceInDifferences`, `InterruptedTimeSeries`, `SyntheticControl`) provide standard methods for retrieving impact estimates.

### `summary(round_to=2)`
Prints a summary of the main results and model coefficients.
*   `round_to`: Number of decimal places to round results to.

### `print_coefficients(round_to=2)`
Prints the coefficients of the underlying model (e.g., regression coefficients).

## Calculation Functions

The `BaseExperiment` and specific experiment classes use these internal calculations:

### `calculate_impact(y, y_hat)`
Calculates the causal impact (lift) by subtracting the counterfactual prediction (`y_hat`) from the observed outcome (`y`).
*   **Formula**: `impact = y - y_hat`
*   **Bayesian Models**: Returns a posterior distribution of the impact.
*   **OLS Models**: Returns point estimates.

### `calculate_cumulative_impact(impact)`
Calculates the cumulative sum of the impact over time. Useful for understanding the total effect of an intervention over a period.

## Plotting Results

Standard plotting methods are available on the experiment objects:

*   `plot()`: Dispatches to `_bayesian_plot` or `_ols_plot` depending on the model type.
*   Returns a `matplotlib` figure and axes.

## Scikit-Learn Compatibility

CausalPy allows using `scikit-learn` estimators via the `SkLearnAdaptor` (or similar wrappers implied by `RegressorMixin` usage).
*   **Pre-Post Fit**: Fits on pre-data, predicts on post-data.
*   **Coefficients**: Standard sklearn `coef_` and `intercept_` are accessed.
