# AGENTS

## Testing preferences

- Write all Python tests as `pytest` style functions, not unittest classes
- Use descriptive function names starting with `test_`
- Prefer fixtures over setup/teardown methods
- Use assert statements directly, not self.assertEqual

## Testing approach

- Never create throwaway test scripts or ad hoc verification files
- If you need to test functionality, write a proper test in the test suite
- All tests go in the `causalpy/tests/` directory following the project structure
- Tests should be runnable with the rest of the suite (`python -m pytest`)
- Even for quick verification, write it as a real test that provides ongoing value
- Preference should be given to integration tests, but unit tests are acceptable for core functionality to maintain high code coverage.
- Tests should remain quick to run. Tests involving MCMC sampling with PyMC should use custom `sample_kwargs` to minimize the computational load.

## Documentation

- **Structure**: Notebooks (how-to examples) go in `docs/source/notebooks/`, knowledgebase (educational content) goes in `docs/source/knowledgebase/`
- **Notebook naming**: Use pattern `{method}_{model}.ipynb` (e.g., `did_pymc.ipynb`, `rd_skl.ipynb`), organized by causal method
- **MyST directives**: Use `:::{note}` and other MyST features for callouts and formatting
- **Glossary linking**: Link to glossary terms (defined in `glossary.rst`) on first mention in a file:
  - In Markdown files (`.md`, `.ipynb`): Use MyST syntax `{term}glossary term``
  - In RST files (`.rst`): Use Sphinx syntax `:term:`glossary term``
- **Cross-references**: For other cross-references in Markdown files, use MyST role syntax with curly braces (e.g., `{doc}path/to/doc`, `{ref}label-name`)
- **Citations**: Use `references.bib` for citations, cite sources in example notebooks where possible. Include reference section at bottom of notebooks using `:::{bibliography}` directive with `:filter: docname in docnames`
- **API documentation**: Auto-generated from docstrings via Sphinx autodoc, no manual API docs needed
- **Build**: Use `make html` to build documentation
- **Doctest**: Use `make doctest` to test that Python examples in doctests work
- **Scratch files**: Put temporary notes and generated markdown in `.scratch/` (untracked). Move anything that should be kept into a tracked location.
- **Markdown formatting**: Do not hard-wrap lines in markdown files; rely on editor auto-wrapping.
- **Issue draft cleanup**: Delete issue draft markdown files from `.scratch/issue_summaries/` after filing.

## Code structure and style

- **Experiment classes**: All experiment classes inherit from `BaseExperiment` in `causalpy/experiments/`. Must declare `supports_ols` and `supports_bayes` class attributes. Only implement abstract methods for supported model types (e.g., if only Bayesian is supported, implement `_bayesian_plot()` and `get_plot_data_bayesian()`; if only OLS is supported, implement `_ols_plot()` and `get_plot_data_ols()`)
- **Model-agnostic design**: Experiment classes should work with both PyMC and scikit-learn models. Use `isinstance(self.model, PyMCModel)` vs `isinstance(self.model, RegressorMixin)` to dispatch to appropriate implementations
- **Model classes**: PyMC models inherit from `PyMCModel` (extends `pm.Model`). Scikit-learn models use `RegressorMixin` and are made compatible via `create_causalpy_compatible_class()`. Common interface: `fit()`, `predict()`, `score()`, `calculate_impact()`, `print_coefficients()`
- **Data handling**: PyMC models use `xarray.DataArray` with coords (keys like "coeffs", "obs_ind", "treated_units"). Scikit-learn models use numpy arrays. Data index should be named "obs_ind"
- **Formulas**: Use patsy for formula parsing (via `dmatrices()`)
- **Custom exceptions**: Use project-specific exceptions from `causalpy.custom_exceptions`: `FormulaException`, `DataException`, `BadIndexException`
- **File organization**: Experiments in `causalpy/experiments/`, PyMC models in `causalpy/pymc_models.py`, scikit-learn models in `causalpy/skl_models.py`

## Code quality checks

- **Before committing**: Always run `pre-commit run --all-files` to ensure all checks pass (linting, formatting, type checking)
- **Quick check**: Run `ruff check causalpy/` for fast linting feedback during development
- **Auto-fix**: Run `ruff check --fix causalpy/` to automatically fix many linting issues
- **Format**: Run `ruff format causalpy/` to format code according to project standards
- **Linting rules**: Project uses strict linting (F, B, UP, C4, SIM, I) to catch bugs and enforce modern Python patterns
- **Note**: Documentation notebooks in `docs/` are excluded from strict linting rules

## Type Checking

- **Tool**: MyPy
- **Configuration**: Integrated as a pre-commit hook.
- **Scope**: Checks Python files within the `causalpy/` directory.
- **Settings**:
    - `ignore-missing-imports`: Enabled to allow for gradual adoption of type hints without requiring all third-party libraries to have stubs.
    - `additional_dependencies`: Includes `numpy` and `pandas-stubs` to provide type information for these libraries.
- **Execution**: Run automatically via `pre-commit run --all-files` or on commit.

## GitHub Issue Workflows

Use the `github-issues` Skill in `.github/skills/github-issues/` for issue
creation, bug reports, and issue evaluation workflows.

## Skills Location

Canonical skills live in `.github/skills/`. The `.claude/skills` and
`.cursor/skills` paths are symlinks to that directory. On Windows, symlink
support may require Developer Mode or elevated permissions; if symlinks are not
available, mirror `.github/skills/` into those locations and keep them in sync.
