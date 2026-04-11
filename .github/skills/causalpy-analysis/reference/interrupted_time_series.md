# Interrupted Time Series

## InterruptedTimeSeries

Analyzes the effect of an intervention on a single time series by comparing pre- and post-intervention trends.

```python
cp.InterruptedTimeSeries(
    data,                    # pd.DataFrame (DatetimeIndex recommended)
    treatment_time,          # int, float, or pd.Timestamp
    formula,                 # str: "y ~ 1 + t + C(month)"
    model=None,              # PyMC or sklearn model
    **kwargs
)
```

**How it works:**
1. Splits data at `treatment_time`
2. Fits model on pre-intervention data only
3. Predicts counterfactual for post-intervention period
4. Impact: observed post minus counterfactual predictions

### Example

```python
import causalpy as cp
import pandas as pd

df = cp.load_data("its")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

result = cp.InterruptedTimeSeries(
    df,
    treatment_time=pd.Timestamp("2017-01-01"),
    formula="y ~ 1 + t + C(month)",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"draws": 1000, "random_seed": 42}
    ),
)
result.plot()
result.effect_summary()
```

### With experimental BSTS models

```python
result = cp.InterruptedTimeSeries(
    df,
    treatment_time=pd.Timestamp("2017-01-01"),
    formula="y ~ 1",
    model=cp.pymc_models.BayesianBasisExpansionTimeSeries(
        n_order=3, n_changepoints_trend=10,
        sample_kwargs={"draws": 1000, "random_seed": 42}
    ),
)
```

## PiecewiseITS

Segmented regression with multiple intervention points using `step()` and `ramp()` transforms.

```python
cp.PiecewiseITS(
    data,
    formula,    # str using step() and ramp(): "y ~ 1 + t + step(t, 50) + ramp(t, 50)"
    model=None,
    **kwargs
)
```

**Transforms:**
- `step(time, threshold)` — level change indicator (0/1)
- `ramp(time, threshold)` — slope change (max(0, time - threshold))
- Both support numeric and datetime thresholds

### Example

```python
from causalpy import step, ramp

result = cp.PiecewiseITS(
    df,
    formula="y ~ 1 + t + step(t, 50) + ramp(t, 50) + step(t, 100) + ramp(t, 100)",
    model=cp.pymc_models.LinearRegression(),
)
result.plot()
```
