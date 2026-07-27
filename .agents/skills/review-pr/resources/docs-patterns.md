# CausalPy Docs Patterns

Use this resource for documentation, notebook, knowledgebase, citation, and MyST review.

## Notebook Structure

New notebooks in `docs/source/notebooks/` should be reachable from the docs tree, usually through `docs/source/notebooks/index.md`.

- ITS examples belong under Interrupted Time Series.
- DiD examples belong under Difference in Differences.
- New method introductions may need a new section.
- Workflow or cross-cutting notebooks may belong under Workflow.

If placement is ambiguous, ask the contributor to confirm rather than guessing.

## Notebook Narrative

Good example notebooks have a clear arc: problem, method, data, analysis, interpretation, caveats.

Review prompts:

- Is the causal question stated before the method is introduced?
- Can a reader who knows causal inference but not CausalPy follow the analysis?
- Are identifying assumptions and caveats visible rather than buried?
- Does the conclusion answer the question posed at the top?
- Are code outputs and prose consistent?

## Cell Tags and Output Noise

Use notebook tags to keep rendered docs readable.

- `hide-input` is useful for plotting cells where matplotlib boilerplate distracts from the figure.
- `hide-output` is useful for sampler progress, long tables, debug prints, or warnings that are not part of the teaching goal.

Visible sampler warnings, divergences, low ESS, or high R-hat values in an educational notebook need attention. Prefer genuine sampler fixes when feasible; hide output only when the warning is acceptable and not central to the narrative.

## External Data

Prefer bundled CausalPy example datasets loaded through `cp.load_data(...)` over runtime fetches from external URLs. If external data is unavoidable, document the dependency and cite the source.

## MyST and Cross-References

Link glossary terms on first mention.

- Markdown and notebooks: `` {term}`glossary term` ``
- RST: `` :term:`glossary term` ``

Use MyST roles for docs cross-references:

- `` {doc}`path/to/doc` ``
- `` {ref}`label-name` ``

Prefer MyST admonitions such as `:::{note}`, `:::{warning}`, and `:::{important}` for callouts instead of plain bold text.

## Citations

Citations live in `docs/source/references.bib`. Example notebooks should include a bibliography block when they cite papers or datasets:

```markdown
:::{bibliography}
:filter: docname in docnames
:::
```

Flag claims or external datasets that need citations but lack bibliography entries.

## Naming and Placement

- Notebook naming pattern: `{method}_{model}.ipynb`, such as `did_pymc.ipynb` or `rd_skl.ipynb`.
- How-to examples belong in `docs/source/notebooks/`.
- Educational and conceptual content belongs in `docs/source/knowledgebase/`.
- Scratch or generated notes belong in `.scratch/`, not tracked docs.

## API Docs

API docs are generated from docstrings via Sphinx autodoc, so public docstrings matter.

- Use NumPy-style sections such as `Parameters`, `Returns`, and `Examples`.
- Examples should be runnable doctests where feasible.
- New public functions with terse or missing examples are usually should-fix, not nits, because the generated API docs inherit that gap.
