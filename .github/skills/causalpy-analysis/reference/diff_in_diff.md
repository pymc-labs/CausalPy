# Difference-in-Differences

## DifferenceInDifferences

Compares trend changes between treated and control groups before and after treatment.

```python
cp.DifferenceInDifferences(
    data,                                      # pd.DataFrame
    formula,                                   # str: "y ~ 1 + group*post_treatment"
    time_variable_name,                        # str: column name for time
    group_variable_name,                       # str: column for group (0=control, 1=treated)
    post_treatment_variable_name="post_treatment",  # str: column for post period (0/1)
    model=None,                                # PyMC or sklearn model
    **kwargs
)
```

**Requirements:**
- Group variable must be dummy coded (0/1)
- Post-treatment variable must be binary (0/1)
- Formula should include group * post_treatment interaction

**How it works:**
1. Fits model on all data (pre/post, treatment/control)
2. Counterfactual: sets interaction term to 0
3. Impact: observed minus counterfactual

### Example

```python
import causalpy as cp

df = cp.load_data("did")
result = cp.DifferenceInDifferences(
    df,
    formula="y ~ 1 + group*post_treatment",
    time_variable_name="t",
    group_variable_name="group",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"draws": 1000, "target_accept": 0.9, "random_seed": 42}
    ),
)
result.plot()
result.effect_summary()
```

## StaggeredDifferenceInDifferences

For staggered treatment adoption across units over time.

```python
cp.StaggeredDifferenceInDifferences(
    data,
    formula,                                   # str
    unit_variable_name,                        # str: unit identifier column
    time_variable_name,                        # str: time column
    treated_variable_name="treated",           # str: treatment indicator column
    treatment_time_variable_name,              # str: when each unit was treated
    never_treated_value,                       # value indicating never-treated units
    model=None,
    **kwargs
)
```

**Requirements:**
- Treatment must be absorbing (irreversible)
- Must have never-treated units for comparison

### Example

```python
df = cp.load_data("did")  # with staggered timing columns
result = cp.StaggeredDifferenceInDifferences(
    df,
    formula="y ~ 1 + treated",
    unit_variable_name="unit",
    time_variable_name="t",
    treatment_time_variable_name="treatment_time",
    never_treated_value=0,
    model=cp.pymc_models.LinearRegression(),
)
result.plot()
```
