# Causal Synthetic Control (SCG)

Synthetic Control constructs a "synthetic" counterfactual unit using a weighted combination of untreated control units.

## Class: `SyntheticControl`

```python
causalpy.experiments.SyntheticControl(
    data,
    treatment_time,
    control_units,
    treated_units,
    model=None,
    **kwargs
)
```

### Parameters
*   **`data`** (`pd.DataFrame`): Input dataframe containing panel data.
*   **`treatment_time`** (`Union[int, float, pd.Timestamp]`): The time of intervention.
*   **`control_units`** (`List[str]`): List of column names representing the control units.
*   **`treated_units`** (`List[str]`): List of column names representing the treated unit(s).
*   **`model`**: A PyMC model (typically `cp.pymc_models.WeightedSumFitter`) or a Scikit-Learn Regressor.

### How it Works
1.  **Fit**: Model learns weights for `control_units` to approximate `treated_units` using **only pre-intervention data**.
2.  **Predict**: Weights are applied to `control_units` in post-intervention period.
3.  **Impact**: Difference between observed treated unit and synthetic counterfactual.

### Example

```python
import causalpy as cp
import causalpy.pymc_models as cp_pymc

df = cp.load_data("sc")
treatment_time = 70

result = cp.SyntheticControl(
    df,
    treatment_time,
    control_units=["a", "b", "c", "d", "e"],
    treated_units=["actual"],
    model=cp_pymc.WeightedSumFitter()
)

result.summary()
result.plot()
```
