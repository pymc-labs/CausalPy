# Causal Interrupted Time Series (ITS)

Interrupted Time Series (ITS) analyzes the effect of an intervention on a single time series by comparing the trend before and after the intervention.

## Class: `InterruptedTimeSeries`

```python
causalpy.experiments.InterruptedTimeSeries(
    data,
    treatment_time,
    formula,
    model=None,
    **kwargs
)
```

### Parameters
*   **`data`** (`pd.DataFrame`): Input dataframe. Index should ideally be a `pd.DatetimeIndex`.
*   **`treatment_time`** (`Union[int, float, pd.Timestamp]`): The point in time when the intervention occurred.
*   **`formula`** (`str`): Statistical formula (e.g., `"y ~ 1 + t + C(month)"`).
*   **`model`**: A PyMC model (e.g., `cp.pymc_models.LinearRegression`) or a Scikit-Learn Regressor.

### How it Works
1.  **Split**: Data is split into pre- and post-intervention.
2.  **Fit**: Model is trained **only on pre-intervention data**.
3.  **Predict**: Fitted model predicts the outcome for the post-intervention period.
4.  **Impact**: Difference between observed post-intervention data and counterfactual predictions.

### Example

```python
import causalpy as cp
import causalpy.pymc_models as cp_pymc
import pandas as pd

df = cp.load_data("its")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

treatment_time = pd.to_datetime("2017-01-01")

result = cp.InterruptedTimeSeries(
    df,
    treatment_time,
    formula="y ~ 1 + t + C(month)",
    model=cp_pymc.LinearRegression()
)

result.summary()
result.plot()
```
