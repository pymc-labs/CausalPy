# Causal Methods

This command provides information and usage examples for the core causal inference methods available in CausalPy:
1. Difference-in-Differences (DiD)
2. Interrupted Time Series (ITS)
3. Synthetic Control (SCG)

For details on how to retrieve estimates, summarize results, and plot outputs after fitting these models, please refer to the [Causal Estimators](causalpy_estimators.md) command (`@causalpy_estimators`).

---

## 1. Causal Difference-in-Differences (DiD)

Difference-in-Differences (DiD) estimates the causal effect of a treatment by comparing the changes in outcomes over time between a treatment group and a control group.

### Class: `DifferenceInDifferences`

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

#### Parameters
*   **`data`** (`pd.DataFrame`): Input dataframe containing panel data.
*   **`formula`** (`str`): Statistical formula (e.g., `"y ~ 1 + group * post_treatment"`).
*   **`time_variable_name`** (`str`): Column name for the time variable.
*   **`group_variable_name`** (`str`): Column name for the group indicator (0=Control, 1=Treated). **Must be dummy coded**.
*   **`post_treatment_variable_name`** (`str`): Column name indicating the post-treatment period (0=Pre, 1=Post). Default is `"post_treatment"`.
*   **`model`**: A PyMC model (e.g., `cp.pymc_models.LinearRegression`) or a Scikit-Learn Regressor.

#### How it Works
1.  **Fit**: The model fits all available data (pre/post, treatment/control).
2.  **Counterfactual**: The counterfactual is predicted by setting the interaction term between `group` and `post_treatment` to 0 (i.e., what would have happened if the treatment group had not been treated in the post-period).
3.  **Impact**: The causal impact is the coefficient of the interaction term (in linear models) or the difference between observed and counterfactual.

#### Example

```python
import causalpy as cp
import causalpy.pymc_models as cp_pymc

# Load data
df = cp.load_data("did")

# Run DiD
result = cp.DifferenceInDifferences(
    df,
    formula="y ~ 1 + group*post_treatment",
    time_variable_name="t",
    group_variable_name="group",
    model=cp_pymc.LinearRegression(sample_kwargs={"target_accept": 0.9})
)

# Summarize
result.summary()

# Plot
result.plot()
```

#### Key Assumptions
*   **Parallel Trends**: Trends in the outcome variable would be the same for both groups in the absence of treatment.

---

## 2. Causal Interrupted Time Series (ITS)

Interrupted Time Series (ITS) analyzes the effect of an intervention on a single time series by comparing the trend before and after the intervention.

### Class: `InterruptedTimeSeries`

```python
causalpy.experiments.InterruptedTimeSeries(
    data,
    treatment_time,
    formula,
    model=None,
    **kwargs
)
```

#### Parameters
*   **`data`** (`pd.DataFrame`): Input dataframe. Index should ideally be a `pd.DatetimeIndex`.
*   **`treatment_time`** (`Union[int, float, pd.Timestamp]`): The point in time when the intervention occurred.
*   **`formula`** (`str`): Statistical formula (e.g., `"y ~ 1 + t + C(month)"`).
*   **`model`**: A PyMC model (e.g., `cp.pymc_models.LinearRegression`, `cp.pymc_models.BayesianBasisExpansionTimeSeries`) or a Scikit-Learn Regressor.

#### How it Works
1.  **Split**: Data is split into pre-intervention and post-intervention sets based on `treatment_time`.
2.  **Fit**: The model is trained **only on the pre-intervention data**.
3.  **Predict**: The fitted model predicts the outcome for the post-intervention period (the counterfactual).
4.  **Impact**: The causal impact is the difference between the observed post-intervention data and the model's counterfactual predictions.

#### Example

```python
import causalpy as cp
import causalpy.pymc_models as cp_pymc
import pandas as pd

# Load data
df = cp.load_data("its")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

treatment_time = pd.to_datetime("2017-01-01")

# Run ITS
result = cp.InterruptedTimeSeries(
    df,
    treatment_time,
    formula="y ~ 1 + t + C(month)",
    model=cp_pymc.LinearRegression()
)

# Summary and Plot
result.summary()
result.plot()
```

#### Key Considerations
*   **Seasonality**: Include seasonal components (e.g., `C(month)`) in the formula if the data exhibits seasonal patterns.
*   **Trends**: Ensure the model captures the underlying trend (e.g., linear time trend `t`) to avoid attributing secular trends to the intervention.

---

## 3. Causal Synthetic Control (SCG)

Synthetic Control constructs a "synthetic" counterfactual unit using a weighted combination of untreated control units that best matches the treated unit's pre-intervention trajectory.

### Class: `SyntheticControl`

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

#### Parameters
*   **`data`** (`pd.DataFrame`): Input dataframe containing panel data.
*   **`treatment_time`** (`Union[int, float, pd.Timestamp]`): The time of intervention.
*   **`control_units`** (`List[str]`): List of column names representing the control units.
*   **`treated_units`** (`List[str]`): List of column names representing the treated unit(s).
*   **`model`**: A PyMC model (typically `cp.pymc_models.WeightedSumFitter`) or a Scikit-Learn Regressor.

#### How it Works
1.  **Fit**: The model learns weights for the `control_units` to approximate the `treated_units` using **only pre-intervention data**.
2.  **Predict**: These weights are applied to the `control_units` in the post-intervention period to generate the synthetic counterfactual.
3.  **Impact**: The difference between the observed treated unit and the synthetic counterfactual.

#### Example

```python
import causalpy as cp
import causalpy.pymc_models as cp_pymc

# Load data
df = cp.load_data("sc")
treatment_time = 70

# Run Synthetic Control
result = cp.SyntheticControl(
    df,
    treatment_time,
    control_units=["a", "b", "c", "d", "e"],
    treated_units=["actual"],
    model=cp_pymc.WeightedSumFitter()
)

# Summary and Plot
result.summary()
result.plot()
```

#### Model Selection
*   **`WeightedSumFitter`**: Enforces that weights sum to 1 and are non-negative (standard Synthetic Control constraints).
*   **`LinearRegression`**: Can be used but allows negative weights and intercept, effectively relaxing the standard SC constraints (sometimes called "Geolift" style or unconstrained SC).
