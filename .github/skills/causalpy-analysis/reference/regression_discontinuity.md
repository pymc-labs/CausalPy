# Regression Discontinuity & Regression Kink

## RegressionDiscontinuity

Estimates causal effect at a threshold in a running variable.

```python
cp.RegressionDiscontinuity(
    data,
    formula,                          # str: "y ~ 1 + x + treated"
    treatment_threshold,              # float: cutoff value
    model=None,                       # PyMC or sklearn
    running_variable_name="x",        # str: running variable column
    epsilon=0.001,                    # float: bandwidth for treated indicator
    bandwidth=np.inf,                 # float: limits data around threshold
    donut_hole=0.0,                   # float: exclude observations near threshold
    **kwargs
)
```

**Supports:** OLS and Bayesian.

### Example

```python
import causalpy as cp

df = cp.load_data("rd")
result = cp.RegressionDiscontinuity(
    df,
    formula="y ~ 1 + x + treated",
    treatment_threshold=0.5,
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"draws": 1000, "random_seed": 42}
    ),
)
result.plot()
result.effect_summary()
```

### With bandwidth and donut hole

```python
result = cp.RegressionDiscontinuity(
    df,
    formula="y ~ 1 + x + treated",
    treatment_threshold=0.5,
    bandwidth=0.3,      # only use data within ±0.3 of threshold
    donut_hole=0.05,    # exclude data within ±0.05 of threshold
    model=cp.pymc_models.LinearRegression(),
)
```

### Recommended checks

- `BandwidthSensitivity` — test sensitivity to bandwidth choice
- `McCraryDensityTest` — check for manipulation at the threshold

## RegressionKink

Estimates causal effect from a slope change (kink) at a threshold. **Bayesian only.**

```python
cp.RegressionKink(
    data,
    formula,                          # str
    kink_point,                       # float: kink location
    model=None,                       # PyMC model only
    running_variable_name="x",
    epsilon=0.001,
    bandwidth=np.inf,
    **kwargs
)
```

### Example

```python
result = cp.RegressionKink(
    df,
    formula="y ~ 1 + x + treated",
    kink_point=0.5,
    model=cp.pymc_models.LinearRegression(),
)
result.plot()
```
