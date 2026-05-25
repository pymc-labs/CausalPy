---
globs: causalpy/**/*.py
---

## Code structure and style

- **Experiment classes**: All experiment classes inherit from `BaseExperiment` in `causalpy/experiments/`. Must declare `supports_ols` and `supports_bayes` class attributes. Only implement abstract methods for supported model types.
- **Model-agnostic design**: Experiment classes should work with both PyMC and scikit-learn models. Use `isinstance(self.model, PyMCModel)` vs `isinstance(self.model, RegressorMixin)` to dispatch.
- **Model classes**: PyMC models inherit from `PyMCModel` (extends `pm.Model`). Scikit-learn models use `RegressorMixin` via `create_causalpy_compatible_class()`. Common interface: `fit()`, `predict()`, `score()`, `calculate_impact()`, `print_coefficients()`.
- **Data handling**: PyMC models use `xarray.DataArray` with coords (keys like "coeffs", "obs_ind", "treated_units"). Scikit-learn models use numpy arrays. Data index should be named "obs_ind".
- **Formulas**: Use patsy for formula parsing (via `dmatrices()`).
- **Custom exceptions**: Use project-specific exceptions from `causalpy.custom_exceptions`: `FormulaException`, `DataException`, `BadIndexException`.
- **File organization**: Experiments in `causalpy/experiments/`, PyMC models in `causalpy/pymc_models.py`, scikit-learn models in `causalpy/skl_models.py`.
- **Backwards compatibility**: Avoid preserving backwards compatibility for API elements introduced within the same PR; only maintain compatibility for previously released APIs.

## Code quality checks

- **Before committing**: Use `prek run` during iterative edits and run `prek run --all-files` before committing.
- **Quick check**: Run `ruff check causalpy/` for fast linting feedback during development.
- **Auto-fix**: Run `ruff check --fix causalpy/` to automatically fix many linting issues.
- **Format**: Run `ruff format causalpy/` to format code according to project standards.
- **Linting rules**: Project uses strict linting (F, B, UP, C4, SIM, I) to catch bugs and enforce modern Python patterns.

## Type checking

- **Tool**: MyPy, integrated as a prek hook.
- **Style**: Use Python 3.10+ type hint syntax: `X | None` not `Optional[X]`, lowercase `dict`, `list`, `tuple` not `Dict`, `List`, `Tuple` from `typing`, and `Literal` for constrained string parameters.
