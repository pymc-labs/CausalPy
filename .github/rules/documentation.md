---
globs: docs/**
---

## Documentation conventions

- **Structure**: Notebooks (how-to examples) go in `docs/source/notebooks/`, knowledgebase (educational content) goes in `docs/source/knowledgebase/`.
- **Notebook naming**: Use pattern `{method}_{model}.ipynb` (e.g., `did_pymc.ipynb`, `rd_skl.ipynb`), organized by causal method.
- **MyST directives**: Use `:::{note}` and other MyST features for callouts and formatting.
- **Glossary linking**: Link to glossary terms (defined in `glossary.rst`) on first mention in a file:
  - In Markdown files (`.md`, `.ipynb`): Use MyST syntax `{term}glossary term``
  - In RST files (`.rst`): Use Sphinx syntax `:term:`glossary term``
- **Cross-references**: For other cross-references in Markdown files, use MyST role syntax with curly braces (e.g., `{doc}path/to/doc`, `{ref}label-name`).
- **Citations**: Use `references.bib` for citations, cite sources in example notebooks where possible. Include reference section at bottom of notebooks using `:::{bibliography}` directive with `:filter: docname in docnames`.
- **API documentation**: Auto-generated from docstrings via Sphinx autodoc, no manual API docs needed.
- **Build**: Use `make html` to build documentation.
- **Doctest**: Use `make doctest` to test that Python examples in doctests work.
- **Notebook schema validation**: `prek run --all-files` runs `validate-notebooks` to catch `.ipynb` files that are valid JSON but invalid nbformat.
- **Notebook validation failure recovery**: Re-open and save (or re-run) in a notebook-aware editor; if it still fails, restore from `main` and reapply intended edits with notebook-aware tooling; rerun `prek run --all-files`.
- **Markdown formatting**: Do not hard-wrap lines in markdown files; rely on editor auto-wrapping.
- **Note**: Documentation notebooks in `docs/` are excluded from strict linting rules.
