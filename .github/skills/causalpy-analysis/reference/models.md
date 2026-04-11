# Models

## PyMC Models (Bayesian)

All inherit from `PyMCModel` which extends `pm.Model`. Common interface: `fit()`, `predict()`, `score()`, `calculate_impact()`, `print_coefficients()`.

### LinearRegression

General-purpose linear regression. Used by DiD, ITS, PiecewiseITS, RD, RK, PrePostNEGD.

```python
import causalpy.pymc_models as cp_pymc
model = cp_pymc.LinearRegression(
    sample_kwargs={"draws": 1000, "target_accept": 0.9, "random_seed": 42}
)
```

Default priors: `Normal(0, 50)` for beta, `HalfNormal(1)` for sigma.

### WeightedSumFitter

Constrained weights for Synthetic Control (weights sum to 1 via Dirichlet).

```python
model = cp_pymc.WeightedSumFitter(
    sample_kwargs={"draws": 2000, "random_seed": 42}
)
```

Default priors: `Dirichlet(1, ..., 1)` for weights (N = number of control units), `HalfNormal(1)` for sigma.

### PropensityScore

Logistic regression for propensity score estimation. Used by IPW.

```python
model = cp_pymc.PropensityScore(
    sample_kwargs={"draws": 1000, "random_seed": 42}
)
```

Default priors: `Normal(0, 1)` for coefficients. Bernoulli likelihood with logit link.

### InstrumentalVariableRegression

Two-stage IV regression with optional variable selection priors.

```python
model = cp_pymc.InstrumentalVariableRegression(
    sample_kwargs={"draws": 2000, "random_seed": 42}
)
```

Supports `vs_prior_type`: `"spike_and_slab"`, `"horseshoe"`, `"normal"`. Supports `binary_treatment=True`.

### BayesianBasisExpansionTimeSeries (EXPERIMENTAL)

Trend + Fourier seasonality for time series. Used by ITS for complex temporal patterns.

```python
model = cp_pymc.BayesianBasisExpansionTimeSeries(
    n_order=3, n_changepoints_trend=10, prior_sigma=5,
    sample_kwargs={"draws": 1000, "random_seed": 42}
)
```

Supports custom `trend_component` and `seasonality_component`, exogenous regressors.

### StateSpaceTimeSeries (EXPERIMENTAL)

Structural time series via pymc-extras state-space models.

```python
model = cp_pymc.StateSpaceTimeSeries(
    level_order=2, seasonal_length=12,
    sample_kwargs={"draws": 1000, "random_seed": 42}
)
```

Supports custom `trend_component` and `seasonality_component`.

## Scikit-Learn Models (OLS)

### Using any sklearn estimator

```python
from sklearn.linear_model import LinearRegression
import causalpy as cp

CausalLR = cp.create_causalpy_compatible_class(LinearRegression)
model = CausalLR(fit_intercept=True)
```

This wraps any `RegressorMixin` with CausalPy methods: `get_coeffs()`, `print_coefficients()`, `calculate_impact()`, `calculate_cumulative_impact()`.

### WeightedProportion

Constrained optimization for OLS synthetic control (weights in [0,1], sum to 1).

```python
from causalpy.skl_models import WeightedProportion
model = WeightedProportion()
```

## Custom Priors (PyMC)

Override default priors via the `priors` parameter:

```python
model = cp_pymc.LinearRegression(
    sample_kwargs={"draws": 1000},
    priors={"beta": "Normal(0, 10)", "sigma": "HalfNormal(2)"}
)
```

Data-driven priors via `priors_from_data()` method for automatic prior scaling.

## Controlling MCMC

Key `sample_kwargs` parameters:
- `draws` (int): Number of posterior draws (default varies)
- `target_accept` (float): Target acceptance rate (0.8-0.99)
- `chains` (int): Number of MCMC chains
- `random_seed` (int): Reproducibility seed
- `cores` (int): Number of parallel cores
