# Data and Dataset PR Review

Use this resource when a PR adds or changes packaged datasets, CSV files, dataset loaders, metadata, examples built around datasets, or generated data assets.

## Review Focus

- Verify dataset provenance, license, citation, and intended teaching or testing purpose are clear.
- Check that no private, personal, credential-like, or sensitive data is included.
- Confirm file names, loader names, docstrings, and examples follow existing CausalPy dataset conventions.
- Inspect schema stability: column names, dtypes, date parsing, index expectations, missing values, units, and treatment/control identifiers.
- Verify loaders use structured parsing through pandas or existing helpers rather than brittle string manipulation.
- Check package inclusion expectations so files needed at runtime are included and unused generated artifacts are not committed.
- Review data size and format for repository bloat, CI impact, and documentation build performance.

## Required Evidence

- New or changed loaders should have tests that check shape, essential columns, index behavior, and representative values.
- Dataset-backed docs or notebooks should cite the source and explain enough context for interpretation.
- If data is transformed from an upstream source, the transformation should be reproducible or clearly documented.

## Review Output Emphasis

- In the PR summary, state what data is added or changed, why it belongs in CausalPy, and how users will access it.
- In findings, foreground provenance, license, privacy, schema stability, loader behavior, package inclusion, and repository size.
- In test evidence, call out loader tests, representative-value checks, and any docs/notebook validation using the dataset.
- In open questions, focus on data source rights, transformation reproducibility, and whether the dataset is minimal for its purpose.

## Request Changes When

- Provenance, license, or citation is missing for external data.
- The dataset contains sensitive data or fields that are not needed for the example.
- Loader tests do not catch schema drift or missing packaged files.
- The data file is much larger than needed for the example or test.
- The dataset encourages an unsupported causal interpretation without explaining assumptions.
