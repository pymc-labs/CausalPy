# CausalPy Code Patterns

Use this resource when reviewing source-code diffs. It collects CausalPy contracts that are important but not always mechanically enforced.

## `BaseExperiment` Contract

All experiment classes in `causalpy/experiments/` inherit from `BaseExperiment`.

- Declare `supports_ols: bool` and `supports_bayes: bool`.
- Implement a single backend-agnostic `_plot()` (and `get_plot_data()` where applicable) that consumes the canonical prediction container.
- Key uncertainty rendering on data properties (`has_posterior_draws()` from `causalpy.plot_utils`), not backend identity; any surviving `is_bayesian` branch needs written justification.
- Keep the data index named `"obs_ind"`.
- Parse formulas through patsy `dmatrices()`.
- Use project exceptions from `causalpy.custom_exceptions` for formula, data, and index errors.

Review prompts:

- Does a new subclass declare the support flags?
- Does support for OLS/Bayesian match the methods actually implemented?
- Does validation use custom exceptions rather than `assert`?
- Does the new class preserve model-agnostic behavior or clearly reject unsupported models?

## `PyMCModel` Contract

PyMC models inherit from `PyMCModel` and share the public interface `fit()`, `predict()`, `score()`, `calculate_impact()`, and `print_coefficients()`.

- Use `xarray.DataArray` with coordinates such as `coeffs`, `obs_ind`, and `treated_units`.
- Store user-supplied priors on `self._user_priors` so they round-trip through `_clone()`.
- Keep sampling configuration explicit and runtime-controlled for tests and examples.

## `_clone()` Pattern

The base `PyMCModel._clone()` forwards `priors=self._user_priors`. Every override must preserve that behavior or user priors are silently dropped.

```python
def _clone(self):
    return type(self)(
        sample_kwargs=self.sample_kwargs,
        priors=self._user_priors,
        # Include any subclass-specific configuration here.
    )
```

If `priors=` is absent from a new override, treat it as a must-fix unless there is a strong reason the model cannot accept custom priors.

## Scikit-Learn Compatibility

Scikit-learn models use `RegressorMixin` and are made CausalPy-compatible through `create_causalpy_compatible_class()`. Experiments should not feature-detect arbitrary methods when the local pattern is type-based dispatch.

Review prompts:

- Does an experiment that claims model-agnostic support branch correctly for PyMC and scikit-learn?
- Does the scikit-learn path use numpy arrays and the PyMC path use xarray-shaped outputs?
- Are unsupported combinations rejected clearly?

## Type Hints

Use Python 3.10+ style:

- `X | None`, not `Optional[X]`.
- Lowercase `dict`, `list`, and `tuple`, not `Dict`, `List`, and `Tuple`.
- `Literal` for constrained string parameters.

Any new parameter typed as `str` but documented or implemented as a closed set of values should usually be a `Literal`.

## Custom Exceptions

Prefer:

- `FormulaException` for patsy/formula problems.
- `DataException` for data shape, missing column, or dtype problems.
- `BadIndexException` for index issues.

Plain `ValueError` is acceptable when no project-specific exception fits. `assert` is not input validation.

## `__repr__` Style

Prefer concise representations that show non-default values only, matching sibling classes.

```python
def __repr__(self) -> str:
    parts = [f"alpha={self.alpha}"] if self.alpha != 0.05 else []
    if not self.store_experiments:
        parts.append("store_experiments=False")
    return f"OutcomeFalsification({', '.join(parts)})"
```

Flag a new `__repr__` that dumps defaults when nearby classes only show meaningful deviations.

## Memory-Heavy Retainers

`InferenceData`, fitted experiments, posterior arrays, and generated figures can be large. New result/check classes should not retain heavy objects by default unless the common user path needs them.

Review prompt: if a result object stores a fitted model, experiment, or posterior by default, ask whether summary-only should be the default with an opt-in such as `store_experiments=True`.

## Backwards Compatibility

Preserve compatibility for previously released APIs. APIs introduced within the same PR can be reshaped freely before merge. Do not add compatibility shims for unshipped API shapes on the current branch.
