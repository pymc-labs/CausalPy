# AGENTS

## Environment

This project uses a conda environment named **CausalPy**. All Python-related commands (`python`, `pytest`, `pre-commit`, `ruff`, `mypy`) must run inside that environment. The env is defined in `environment.yml` at the project root.

### When to use what

- **Codex desktop app, git worktrees, or fresh shells** (conda not auto-activated):
  **Always** use `conda run -n CausalPy <command>`. Do not rely on activation; the shell may not have conda in PATH or the env active. Example: `conda run -n CausalPy python -m pytest causalpy/tests/`

- **Cursor or other IDE where conda is available and you can activate:**
  Either activate once then run commands normally, or use `conda run -n CausalPy ...` for a single command. Prefer `conda run` when suggesting commands for the user to run in a terminal that might not have the env activated.

- **First-time setup or the CausalPy env does not exist:**
  From the repo root, run `scripts/codex_setup.sh`. It creates the env and does an editable install with `[dev]` extras.

### How to activate (when activation is possible)

```bash
source ~/mambaforge/etc/profile.d/conda.sh && conda activate CausalPy
```

If that path does not exist, use:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate CausalPy
```

### Run without activating (recommended when in doubt)

Use this form for any Python or tooling command when you are not sure the env is active (e.g. in Codex, worktrees, or when suggesting a one-off command):

```bash
conda run -n CausalPy python -m pytest causalpy/tests/
conda run -n CausalPy python -c "import causalpy; print(causalpy.__version__)"
conda run -n CausalPy pre-commit run --all-files
```

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

## Sandbox and permissions

- **PyMC/PyTensor require filesystem access** outside the workspace (e.g. `~/.pytensor/compiledir_*` for C compilation cache, `~/.matplotlib` for font cache). The default Cursor sandbox blocks writes to these paths, causing misleading `ValueError` or `PermissionError` failures that look like real test errors but are not.
- **Always use `required_permissions: ["all"]`** when running `pytest`, `make doctest`, or any command that imports PyMC, PyTensor, or matplotlib. This avoids false negatives from sandbox restrictions.
- **If a test run shows `compiledir ... you don't have read, write or listing permissions`**, that is a sandbox problem, not a code problem. Re-run with `["all"]` permissions before investigating further.

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
  - **PR drafts**: Create PR summary markdown files in `.scratch/pr_summaries/` (untracked).
  - **Issue drafts**: Create issue draft markdown files in `.scratch/issue_summaries/` (untracked).
- **Markdown formatting**: Do not hard-wrap lines in markdown files; rely on editor auto-wrapping.

## Code structure and style

- **Experiment classes**: All experiment classes inherit from `BaseExperiment` in `causalpy/experiments/`. Must declare `supports_ols` and `supports_bayes` class attributes. Only implement abstract methods for supported model types (e.g., if only Bayesian is supported, implement `_bayesian_plot()` and `get_plot_data_bayesian()`; if only OLS is supported, implement `_ols_plot()` and `get_plot_data_ols()`)
- **Model-agnostic design**: Experiment classes should work with both PyMC and scikit-learn models. Use `isinstance(self.model, PyMCModel)` vs `isinstance(self.model, RegressorMixin)` to dispatch to appropriate implementations
- **Model classes**: PyMC models inherit from `PyMCModel` (extends `pm.Model`). Scikit-learn models use `RegressorMixin` and are made compatible via `create_causalpy_compatible_class()`. Common interface: `fit()`, `predict()`, `score()`, `calculate_impact()`, `print_coefficients()`
- **Data handling**: PyMC models use `xarray.DataArray` with coords (keys like "coeffs", "obs_ind", "treated_units"). Scikit-learn models use numpy arrays. Data index should be named "obs_ind"
- **Formulas**: Use patsy for formula parsing (via `dmatrices()`)
- **Custom exceptions**: Use project-specific exceptions from `causalpy.custom_exceptions`: `FormulaException`, `DataException`, `BadIndexException`
- **File organization**: Experiments in `causalpy/experiments/`, PyMC models in `causalpy/pymc_models.py`, scikit-learn models in `causalpy/skl_models.py`
- **Backwards compatibility**: Avoid preserving backwards compatibility for API elements introduced within the same PR; only maintain compatibility for previously released APIs.

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

## GitHub CLI

Use `gh` CLI as the preferred source of truth for GitHub issues, PRs, releases. See [`.github/skills/github-cli/SKILL.md`](.github/skills/github-cli/SKILL.md) for details.

## Skills Location

Canonical skills live in `.github/skills/`. The `.claude/skills` and `.cursor/skills` paths are symlinks to that directory. On Windows, symlink support may require Developer Mode or elevated permissions; if symlinks are not available, mirror `.github/skills/` into those locations and keep them in sync.
