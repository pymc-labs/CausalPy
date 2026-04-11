# IPW, Instrumental Variables & PrePostNEGD

## InversePropensityWeighting

Estimates causal effects by reweighting observations based on propensity scores. **Bayesian only.**

```python
cp.InversePropensityWeighting(
    data,
    formula,                    # str: propensity model formula
    outcome_variable,           # str: outcome column name
    model=None,                 # typically PropensityScore
    weighting_scheme="robust",  # "raw", "robust", "doubly robust", "overlap"
    **kwargs
)
```

**Weighting schemes:**
- `"raw"` — standard IPW
- `"robust"` — normalized weights (default)
- `"doubly robust"` — combines IPW with outcome modeling
- `"overlap"` — overlap weights for better efficiency

### Example

```python
import causalpy as cp

df = cp.load_data("nhefs")
result = cp.InversePropensityWeighting(
    df,
    formula="treatment ~ age + education + smokeintensity",
    outcome_variable="outcome",
    weighting_scheme="robust",
    model=cp.pymc_models.PropensityScore(
        sample_kwargs={"draws": 1000, "random_seed": 42}
    ),
)
result.plot()
result.effect_summary()
```

## InstrumentalVariable

Two-stage regression using instruments to handle endogeneity. **Bayesian only.**

```python
cp.InstrumentalVariable(
    instruments_data,            # pd.DataFrame: instrument data
    data,                        # pd.DataFrame: outcome data
    instruments_formula,         # str: first-stage formula
    formula,                     # str: second-stage formula
    model=None,                  # typically InstrumentalVariableRegression
    priors=None,                 # dict: custom priors
    vs_prior_type=None,          # "spike_and_slab", "horseshoe", "normal"
    vs_hyperparams=None,         # dict: variable selection hyperparameters
    binary_treatment=False,      # bool: binary treatment indicator
    **kwargs
)
```

**Variable selection priors** for identifying relevant instruments:
- `"spike_and_slab"` — discrete spike-and-slab
- `"horseshoe"` — continuous shrinkage

### Example

```python
df = cp.load_data("risk")
result = cp.InstrumentalVariable(
    instruments_data=df[["instrument_col"]],
    data=df,
    instruments_formula="treatment ~ instrument_col",
    formula="outcome ~ treatment",
    model=cp.pymc_models.InstrumentalVariableRegression(
        sample_kwargs={"draws": 2000, "random_seed": 42}
    ),
)
result.plot()
```

### With variable selection

```python
result = cp.InstrumentalVariable(
    instruments_data=df[instrument_cols],
    data=df,
    instruments_formula="treatment ~ " + " + ".join(instrument_cols),
    formula="outcome ~ treatment",
    vs_prior_type="horseshoe",
    model=cp.pymc_models.InstrumentalVariableRegression(),
)
```

## PrePostNEGD

Pre/post analysis with non-equivalent groups design.

```python
cp.PrePostNEGD(
    data,
    formula,                        # str
    group_variable_name,            # str: group column
    pretreatment_variable_name,     # str: pre-treatment outcome column
    model=None,                     # PyMC or sklearn
    **kwargs
)
```

### Example

```python
df = cp.load_data("anova1")
result = cp.PrePostNEGD(
    df,
    formula="post ~ 1 + pre + group",
    group_variable_name="group",
    pretreatment_variable_name="pre",
    model=cp.pymc_models.LinearRegression(),
)
result.plot()
```
