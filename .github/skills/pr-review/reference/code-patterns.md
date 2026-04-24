# CausalPy Code Patterns

Repo-specific conventions and contracts to check against when reviewing source-code diffs. These are the patterns most likely to be inadvertently broken by a contributor (especially a new one) because they're enforced by convention rather than by the type system.

## `BaseExperiment` contract

All experiment classes in `causalpy/experiments/` inherit from `BaseExperiment`. Required:

- Class attributes `supports_ols: bool` and `supports_bayes: bool` declared.
- If `supports_bayes`: implement `_bayesian_plot()` and `get_plot_data_bayesian()`.
- If `supports_ols`: implement `_ols_plot()` and `get_plot_data_ols()`.
- Model-agnostic dispatch via `isinstance(self.model, PyMCModel)` vs. `isinstance(self.model, RegressorMixin)`.
- Data index named `"obs_ind"`.
- Formula parsing via patsy `dmatrices()`.
- Custom exceptions from `causalpy.custom_exceptions`: `FormulaException`, `DataException`, `BadIndexException`.

Review prompts:

- New `BaseExperiment` subclass? Check both class attributes are declared.
- Implements only `_bayesian_plot` but `supports_ols=True`? That's a bug.
- Uses `assert` for input validation? Should be a custom exception.

## `PyMCModel` contract

PyMC models inherit from `PyMCModel` (extends `pm.Model`). Common interface: `fit()`, `predict()`, `score()`, `calculate_impact()`, `print_coefficients()`.

- Data: `xarray.DataArray` with coords (`coeffs`, `obs_ind`, `treated_units`).
- Stores user-supplied priors on `self._user_priors` for round-trip via `_clone()`.

## `_clone()` pattern

The base `PyMCModel._clone()` forwards `priors=self._user_priors`. Every override **must** do the same, or user customisations are silently dropped on clone.

Spot-check template:

```python
# In the new subclass:
def _clone(self):
    return type(self)(
        sample_kwargs=self.sample_kwargs,
        priors=self._user_priors,   # <-- must be present
        # ... any other config kwargs the subclass adds ...
    )
```

If `priors=` is absent, that's a must-fix (MF-1 in [what-to-look-for.md](what-to-look-for.md)).

## `RegressorMixin` (sklearn) compatibility

Scikit-learn models use `RegressorMixin` and are made causalpy-compatible via `create_causalpy_compatible_class()`. Same external interface as `PyMCModel`. Data is numpy arrays, not xarray.

Review prompt: experiment classes should branch with `isinstance(self.model, PyMCModel)` vs. `isinstance(self.model, RegressorMixin)` rather than feature-detect on the model.

## `CheckResult` contract

Lives in `causalpy/checks/base.py`. Has a documented `figures: list[Any]` field — but as of writing, no check populates it. If a new check defines a non-trivial plotting helper in a notebook (see [what-to-look-for.md § N-7](what-to-look-for.md#n-7-helper-used-n-times-in-a-notebook-is-library-material)), promoting it to populate `CheckResult.figures` is the natural integration point.

`passed=None` is the right design for informational-only checks (e.g. `OutcomeFalsification`) — they should inform, not gate.

## Type hints

`AGENTS.md` is explicit. Required style (Python 3.10+):

- `X | None`, not `Optional[X]`.
- Lowercase `dict`, `list`, `tuple`, not `Dict`, `List`, `Tuple` from `typing`.
- `Literal` for constrained string parameters (see [what-to-look-for.md § MF-2](what-to-look-for.md#mf-2-constrained-string-parameter-not-typed-as-literal)).

## Custom exceptions

Use, in order of preference:

- `FormulaException` — patsy/formula problems.
- `DataException` — data-shape, missing-column, dtype problems.
- `BadIndexException` — index issues.

Plain `ValueError` is acceptable when nothing more specific fits; `assert` is not.

## `__repr__` style

Convention: show non-default values only.

```python
def __repr__(self) -> str:
    parts = [f"alpha={self.alpha}"] if self.alpha != 0.05 else []
    if not self.store_experiments:
        parts.append("store_experiments=False")
    return f"OutcomeFalsification({', '.join(parts)})"
```

A new class with a `__repr__` that shows defaults is inconsistent with sibling classes — flag as N-1.

## Helper promotion

When a docs notebook defines a non-trivial helper (≥50 lines) called ≥3 times, suggest promoting to the library. Standard cleanup checklist before promotion:

1. Drop UI side-effects: no `print()` in library code (use `warnings.warn`); no `plt.show()` (return the `Figure`).
2. Drop hard-coded constants that limit reuse: e.g. `FOLD_COLORS = [...5 colors...]` should become a module-level constant or use matplotlib's color cycle.
3. Add `figsize` and `axes` kwargs so the function is composable in user code and unit-testable.
4. Reduce arg surface: if two args are always paired, fold one into the other (or store the dependency on the producing class so the consumer takes one arg, not two).
5. Add a tiny test that asserts shape: `assert isinstance(out, Figure)`, `assert len(out.axes) == expected`.

## Memory-heavy retainers

`InferenceData` is large. Any class that retains it by default imposes a hidden cost. Default to summary-only with an opt-in (`store_X=True`) for users who want to inspect.

## Backwards compatibility

`AGENTS.md`: maintain compatibility for previously-released APIs only; APIs introduced in the same PR can be reshaped freely. So new public APIs (added in the PR being reviewed) are legitimate review targets for "I'd rather it look like X" feedback; APIs that already shipped are not.
