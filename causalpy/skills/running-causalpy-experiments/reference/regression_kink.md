# Regression Kink

Use `cp.RegressionKink` when treatment intensity or incentives change slope at a known kink point in a running variable.

## Constructor

```python
cp.RegressionKink(
    data,
    formula,
    kink_point,
    model=None,
    running_variable_name="x",
    epsilon=0.001,
    bandwidth=np.inf,
    **kwargs
)
```

## Required Data

- `data`: DataFrame with outcome, running variable, and formula terms needed to model the kink.
- `kink_point`: threshold where the slope changes.
- `running_variable_name`: running variable column, defaulting to `"x"`.
- `bandwidth`: optional local window around the kink.

## Model Guidance

Only Bayesian PyMC models are supported. The default is `cp.pymc_models.LinearRegression`. Scale the running variable around the kink and use priors that make the implied slope changes plausible in the outcome units.

## Example

```python
import causalpy as cp

result = cp.RegressionKink(
    data=df,
    formula="y ~ 1 + x + kink",
    kink_point=0.0,
    running_variable_name="x",
    bandwidth=1.0,
    model=cp.pymc_models.LinearRegression(sample_kwargs={"target_accept": 0.95}),
)

result.summary()
summary = result.effect_summary(direction="increase")
result.plot()
```

## Interpretation Checks

- Use `BandwidthSensitivity` to assess whether the estimate depends on the local window.
- Use `RegressionDiscontinuity` instead if the outcome or treatment probability jumps at the cutoff.
- Check that the running variable is not manipulated near the kink.
