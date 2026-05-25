# Instrumental Variable

Use `cp.InstrumentalVariable` when treatment is endogenous and there is a credible instrument for treatment.

## Constructor

```python
cp.InstrumentalVariable(
    instruments_data,
    data,
    instruments_formula,
    formula,
    model=None,
    priors=None,
    vs_prior_type=None,
    vs_hyperparams=None,
    binary_treatment=False,
    **kwargs
)
```

## Required Data

- `instruments_data`: DataFrame containing the treatment variable and instrument columns.
- `data`: DataFrame containing the outcome, treatment, and outcome-stage covariates.
- `instruments_formula`: first-stage formula, for example `"t ~ 1 + z"`.
- `formula`: outcome-stage formula, for example `"y ~ 1 + t + x"`.
- `binary_treatment`: set to `True` for binary endogenous treatment.

## Model Guidance

Only Bayesian PyMC models are supported. The default is `cp.pymc_models.InstrumentalVariableRegression`. Priors are passed to `cp.InstrumentalVariable(..., priors=...)`, not to the model constructor, and should match the scale of the treatment-stage and outcome-stage regressions. Variable-selection priors are available through `vs_prior_type` and `vs_hyperparams`.

## Example

```python
import causalpy as cp

result = cp.InstrumentalVariable(
    instruments_data=df[["t", "z"]],
    data=df[["y", "t", "x"]],
    instruments_formula="t ~ 1 + z",
    formula="y ~ 1 + t + x",
    model=cp.pymc_models.InstrumentalVariableRegression(
        sample_kwargs={"target_accept": 0.95}
    ),
    priors={
        "mus": [[0, 0], [0, 0, 0]],
        "sigmas": [2, 5],
        "eta": 2,
        "lkj_sd": 1,
    },
)
```

## Interpretation Checks

- `plot()`, `summary()`, and `effect_summary()` are not implemented for this experiment.
- Check instrument relevance, exclusion restriction plausibility, and weak-instrument sensitivity before using causal language.
- Use variable-selection priors cautiously and report whether conclusions depend on them.
