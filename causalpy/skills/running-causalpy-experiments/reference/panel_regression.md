# Panel Regression

Use `cp.PanelRegression` when the analysis target is a panel regression with fixed effects rather than a more specialized treatment-timing design.

## Constructor

```python
cp.PanelRegression(
    data,
    formula,
    unit_fe_variable,
    time_fe_variable=None,
    fe_method="dummies",
    model=None,
    **kwargs
)
```

## Required Data

- `data`: long panel where each row is a unit-time observation.
- `unit_fe_variable`: unit identifier column.
- `time_fe_variable`: optional time identifier column.
- `formula`: with `fe_method="dummies"`, include `C(unit)` and optionally `C(time)` yourself; with `fe_method="demeaned"`, do not include those fixed-effect terms.

## Model Guidance

Unlike many other experiments, `PanelRegression` does not set a default model class, so pass either `cp.pymc_models.LinearRegression(...)` or a sklearn-compatible regressor. Use `fe_method="dummies"` for smaller panels where unit effects should be estimated, and `fe_method="demeaned"` for larger panels where dummy expansion is too expensive.

## Example

```python
import causalpy as cp

result = cp.PanelRegression(
    data=df,
    formula="y ~ C(unit) + C(time) + treatment + x1",
    unit_fe_variable="unit",
    time_fe_variable="time",
    fe_method="dummies",
    model=cp.pymc_models.LinearRegression(sample_kwargs={"target_accept": 0.95}),
)

result.summary()
summary = result.effect_summary(direction="increase")
result.plot()
```

## Interpretation Checks

- Treat coefficients as adjusted regression estimates unless the treatment assignment story supports causal interpretation.
- For staggered adoption and event-time effects, prefer `StaggeredDifferenceInDifferences`.
- For heavily unbalanced panels with two-way demeaning, be cautious because the built-in demeaned transformation is a single-pass approximation.
