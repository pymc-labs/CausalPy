# Causal Regression Discontinuity (RD)

Regression Discontinuity exploits a cutoff or threshold in an assignment variable to identify causal effects. Units just above and below the threshold are compared to estimate the treatment effect at the discontinuity.

## Class: `RegressionDiscontinuity`

```python
causalpy.experiments.RegressionDiscontinuity(
    data,
    formula,
    treatment_threshold,
    model=None,
    running_variable_name="x",
    epsilon=0.001,
    bandwidth=np.inf,
    donut_hole=0.0,
    **kwargs
)
```

### Parameters

*   **`data`** (`pd.DataFrame`): Input dataframe.
*   **`formula`** (`str`): Statistical formula (e.g., `"y ~ 1 + x + treated + x:treated"`).
*   **`treatment_threshold`** (`float`): The cutoff value of the running variable where treatment is assigned.
*   **`model`**: A PyMC model (e.g., `cp.pymc_models.LinearRegression`) or a Scikit-Learn Regressor.
*   **`running_variable_name`** (`str`): Column name of the running variable. Default is `"x"`.
*   **`epsilon`** (`float`): Small offset above/below the threshold for evaluating the causal impact. Default is `0.001`.
*   **`bandwidth`** (`float`): Data outside this distance from the threshold is excluded from fitting. Default is `np.inf` (use all data).
*   **`donut_hole`** (`float`): Observations within this distance from the threshold are excluded from fitting (robustness check). Default is `0.0`.

### How it Works

1.  **Fit**: Model is trained on data within the bandwidth, optionally excluding the donut hole.
2.  **Predict**: Counterfactual predicted at the threshold by evaluating the model just above and just below.
3.  **Impact**: The causal effect is the discontinuous jump in the outcome at the threshold.

### Example

```python
import causalpy as cp
import causalpy.pymc_models as cp_pymc

df = cp.load_data("drinking")
df = df.rename(columns={"agecell": "age"}).assign(treated=lambda d: d.age > 21)

result = cp.RegressionDiscontinuity(
    df,
    formula="all ~ 1 + age + treated + age:treated",
    running_variable_name="age",
    model=cp_pymc.LinearRegression(),
    treatment_threshold=21,
)

result.summary()
result.plot()
```
