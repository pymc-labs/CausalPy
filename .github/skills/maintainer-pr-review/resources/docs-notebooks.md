# Docs and Notebook PR Review

Use this resource when a PR changes documentation pages, example notebooks, knowledgebase content, citations, glossary links, images, or narrative explanations.

## Review Focus

- Check that the content is technically accurate, pedagogically clear, and aligned with CausalPy's supported APIs.
- Verify causal claims distinguish identification assumptions from model output and do not imply causality from association alone.
- Confirm first mentions of glossary terms use the correct MyST or Sphinx role syntax.
- Check citations are present where claims or datasets need support and that references are added to `docs/source/references.bib` when needed.
- Verify notebooks are named and placed according to CausalPy conventions, with examples in `docs/source/notebooks/` and educational material in `docs/source/knowledgebase/`.
- Review code cells for runnable imports, deterministic seeds where useful, bounded sampling/runtime, and no stale outputs that contradict the code.
- Check markdown prose is not hard-wrapped unless inside code blocks or structured content.

## Required Evidence

- For notebook changes, `validate-notebooks` through `prek run --all-files` or an equivalent notebook-aware validation should pass before merge.
- For substantial docs changes, `make html` should be run when feasible, especially when cross-references, citations, notebooks, or generated outputs changed.
- Examples that import CausalPy should be checked in the `CausalPy` environment.

## Request Changes When

- The docs teach a workflow that is not executable with the current code.
- Citations, glossary links, cross-references, or bibliography entries are missing or broken.
- The notebook relies on large unbounded sampling or nondeterministic behavior without justification.
- The explanation overstates causal identification, model certainty, or generality of an example.
- Generated data, images, or notebook outputs obscure the review or introduce reproducibility risk.
