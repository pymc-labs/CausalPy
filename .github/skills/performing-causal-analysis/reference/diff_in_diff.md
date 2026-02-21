# Causal Difference-in-Differences (DiD)

Difference-in-Differences (DiD) estimates the causal effect of a treatment by comparing the changes in outcomes over time between a treatment group and a control group.

## Class: `DifferenceInDifferences`

```python
causalpy.experiments.DifferenceInDifferences(
    data,
    formula,
    time_variable_name,
    group_variable_name,
    post_treatment_variable_name="post_treatment",
    model=None,
    **kwargs
)
```

### Parameters
*   **`data`** (`pd.DataFrame`): Input dataframe containing panel data.
*   **`formula`** (`str`): Statistical formula (e.g., `"y ~ 1 + group * post_treatment"`).
*   **`time_variable_name`** (`str`): Column name for the time variable.
*   **`group_variable_name`** (`str`): Column name for the group indicator (0=Control, 1=Treated). **Must be dummy coded**.
*   **`post_treatment_variable_name`** (`str`): Column name indicating the post-treatment period (0=Pre, 1=Post). Default is `"post_treatment"`.
*   **`model`**: A PyMC model (e.g., `cp.pymc_models.LinearRegression`) or a Scikit-Learn Regressor.

### How it Works
1.  **Fit**: The model fits all available data (pre/post, treatment/control).
2.  **Counterfactual**: Predicted by setting the interaction term between `group` and `post_treatment` to 0.
3.  **Impact**: The causal impact is the difference between observed and counterfactual.

### Example

```python
import causalpy as cp
import causalpy.pymc_models as cp_pymc

df = cp.load_data("did")

result = cp.DifferenceInDifferences(
    df,
    formula="y ~ 1 + group*post_treatment",
    time_variable_name="t",
    group_variable_name="group",
    model=cp_pymc.LinearRegression(sample_kwargs={"target_accept": 0.9})
)

result.summary()
result.plot()
```
