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
- **Glossary linking**: Use Sphinx `:term:` directives to link to glossary terms (defined in `glossary.rst`), typically on first mention in a file
- **Citations**: Use `references.bib` for citations, cite sources in example notebooks where possible. Include reference section at bottom of notebooks using `:::{bibliography}` directive with `:filter: docname in docnames`
- **API documentation**: Auto-generated from docstrings via Sphinx autodoc, no manual API docs needed
- **Build**: Use `make html` to build documentation
- **Doctest**: Use `make doctest` to test that Python examples in doctests work
