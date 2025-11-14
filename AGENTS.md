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

### Adding new notebooks to the gallery

When creating a new example notebook:

1. **Place it** in `docs/source/notebooks/` with naming pattern `{method}_{model}.ipynb`
2. **Include at least one plot** in the notebook outputs (the first PNG image will be used as the thumbnail)
3. **Manually add it to `docs/source/notebooks/index.md`**:
   - Find the appropriate category section or create a new one
   - Add a `grid-item-card` entry with:
     - `:img-top: ../_static/thumbnails/{notebook_name}.png` (thumbnail path)
     - `:link: {notebook_name_without_extension}` (notebook name without `.ipynb`)
     - `:link-type: doc`
   - Cards are arranged in 3-column grids using `sphinx-design`
4. **Thumbnails are generated automatically** during the build process by `scripts/generate_gallery.py` (runs via `conf.py` during Sphinx setup)
5. **Test locally** with `make html` and check `docs/_build/html/notebooks/index.html`

**Important**: The `index.md` file is manually maintained. The `generate_gallery.py` script only generates thumbnails; it does not modify `index.md`. Thumbnails are gitignored (`docs/source/_static/thumbnails/`) and generated on-demand during builds.

## Code structure and style

- **Experiment classes**: All experiment classes inherit from `BaseExperiment` in `causalpy/experiments/`. Must declare `supports_ols` and `supports_bayes` class attributes. Only implement abstract methods for supported model types (e.g., if only Bayesian is supported, implement `_bayesian_plot()` and `get_plot_data_bayesian()`; if only OLS is supported, implement `_ols_plot()` and `get_plot_data_ols()`)
- **Model-agnostic design**: Experiment classes should work with both PyMC and scikit-learn models. Use `isinstance(self.model, PyMCModel)` vs `isinstance(self.model, RegressorMixin)` to dispatch to appropriate implementations
- **Model classes**: PyMC models inherit from `PyMCModel` (extends `pm.Model`). Scikit-learn models use `RegressorMixin` and are made compatible via `create_causalpy_compatible_class()`. Common interface: `fit()`, `predict()`, `score()`, `calculate_impact()`, `print_coefficients()`
- **Data handling**: PyMC models use `xarray.DataArray` with coords (keys like "coeffs", "obs_ind", "treated_units"). Scikit-learn models use numpy arrays. Data index should be named "obs_ind"
- **Formulas**: Use patsy for formula parsing (via `dmatrices()`)
- **Custom exceptions**: Use project-specific exceptions from `causalpy.custom_exceptions`: `FormulaException`, `DataException`, `BadIndexException`
- **File organization**: Experiments in `causalpy/experiments/`, PyMC models in `causalpy/pymc_models.py`, scikit-learn models in `causalpy/skl_models.py`

## Type Checking

- **Tool**: MyPy
- **Configuration**: Integrated as a pre-commit hook.
- **Scope**: Checks Python files within the `causalpy/` directory.
- **Settings**:
    - `ignore-missing-imports`: Enabled to allow for gradual adoption of type hints without requiring all third-party libraries to have stubs.
    - `additional_dependencies`: Includes `numpy` and `pandas-stubs` to provide type information for these libraries.
- **Execution**: Run automatically via `pre-commit run --all-files` or on commit.
