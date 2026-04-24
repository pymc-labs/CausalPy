# CausalPy Docs Patterns

Repo-specific conventions for notebooks (`docs/source/notebooks/`) and knowledgebase pages (`docs/source/knowledgebase/`). Mechanical rules are enforced by `prek` / CI hooks; this file is for the judgement-heavy patterns a hook can't catch.

## Notebook structure

### `toctree` placement

A new notebook in `docs/source/notebooks/` must be referenced from somewhere — typically `docs/source/notebooks/index.md`. Cross-link from related prose pages (e.g. `sensitivity_checks.md`) where applicable.

Section assignment is a curation question:

- ITS examples → `Interrupted Time Series` toctree.
- DiD examples → `Difference in Differences`.
- New method introduction → its own toctree section.
- Workflow / cross-cutting → `Workflow` section.

If the placement isn't obvious, ask the contributor to confirm rather than guessing.

## Notebook content

### Pedagogical structure

Good notebooks have a recognisable arc: problem → method → data → analysis → interpretation → caveats. Reviews should call out when the arc is missing or scrambled.

When evaluating pedagogy, useful prompts:

- Can a reader who knows causal inference but not CausalPy follow it?
- Is the *causal question* stated before the method is introduced?
- Are caveats and assumptions called out, not buried?
- Does the conclusion answer the question posed at the top?

### Cell tags

Two tags matter for readability:

- **`hide-input`** on plotting cells — collapses the matplotlib boilerplate so the rendered docs show the plot output prominently.
- **`hide-output`** on cells with verbose output (sampler progress, large tables, debug prints) — collapses the noise while keeping the cell runnable.

Review prompt: any cell that produces ≥10 lines of sampler/progress output without `hide-output` is a candidate for the tag.

### Sampler warnings

Visible sampler warnings (divergences, low ESS, R̂ > 1.01) in a docs notebook are confusing for readers. Two fixes:

1. Tune the sampler config (more chains, more tuning steps) to make the warnings go away genuinely.
2. Hide them under `hide-output` if the warnings aren't a core part of the narrative.

(1) is preferred when feasible. (2) is acceptable when the analysis is fundamentally about a borderline case.

### External data

Notebooks should prefer bundled CausalPy example datasets (loaded via `cp.load_data(...)`) over runtime fetches from external URLs. If an external fetch is unavoidable, document it with a comment so readers know what they're depending on.

## MyST and cross-references

### Glossary linking

`AGENTS.md`: link to glossary terms (defined in `glossary.rst`) on first mention in a file.

- Markdown / `.ipynb`: `` {term}`glossary term` ``
- RST: `` :term:`glossary term` ``

If a notebook introduces concepts like "treatment effect", "potential outcomes", "propensity score", etc., the first mention should be a glossary link. Subsequent mentions don't need to link.

### Cross-references

In Markdown, use MyST role syntax:

- `` {doc}`path/to/doc` `` — link to another docs page.
- `` {ref}`label-name` `` — link to a labelled section.

Don't use raw markdown links (`[text](path/to/doc.md)`) for cross-doc references — MyST roles get full Sphinx resolution and break-link detection.

### Admonitions

Use MyST `:::{note}`, `:::{warning}`, `:::{important}` for callouts. Plain bold/italic emphasis is not the same — admonitions render as styled boxes.

### Citations

Citations live in `docs/source/references.bib`. Cite sources in example notebooks where possible. Include a reference section at the bottom of notebooks using:

```markdown
:::{bibliography}
:filter: docname in docnames
:::
```

If a new notebook has prose claims like "as shown in Smith et al. (2020)" without a corresponding `references.bib` entry and `:::{bibliography}` block, flag it.

## Naming and organisation

### Notebook filenames

Pattern: `{method}_{model}.ipynb`.

- `did_pymc.ipynb`, `rd_skl.ipynb`, `iv_pymc.ipynb`, `sc_pymc_brexit.ipynb`.
- Composite topics: keep the prefix scheme so `:glob:` patterns would still work even though no section uses one today.

### File placement

- How-to examples → `docs/source/notebooks/`.
- Educational / conceptual content → `docs/source/knowledgebase/`.
- Scratch / temporary → `.scratch/` (untracked).

A notebook that's mostly explaining a concept (rather than demonstrating CausalPy usage) probably belongs in `knowledgebase/`, not `notebooks/`.

## API docs

Auto-generated from docstrings via Sphinx autodoc. No manual API docs to write — but the docstrings themselves matter:

- NumPy-style sections (`Parameters`, `Returns`, `Examples`).
- `Examples` blocks should be runnable doctests where feasible (`make doctest` in CI).

If a new public function has no `Examples` block, that's a should-fix, not a nit, because the API docs page will be terse.
