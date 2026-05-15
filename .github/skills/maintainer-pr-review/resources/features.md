# Feature PR Review

Use this resource when a PR adds a new user-facing capability, experiment option, model, plotting behavior, loader, public helper, or documented workflow.

## Review Focus

- Confirm the feature's user story is clear from the PR title, description, tests, docs, or code.
- Check that new behavior matches existing CausalPy patterns for experiments, PyMC models, scikit-learn compatibility, formulas, data handling, plotting, and exceptions.
- Verify public APIs have explicit signatures, documented parameters, stable return types, and no bare `*args` or `**kwargs` at public plotting surfaces.
- Check that model-agnostic features work for every supported backend or explicitly reject unsupported model types with project exceptions.
- Review edge cases: empty data, missing values, nonstandard indexes, multiple treated units, coordinate names, posterior dimensions, and formula parsing.
- Check that new examples and docs make causal assumptions clear and avoid overstating identification or interpretation.

## Required Evidence

- Tests should cover the main happy path and at least one meaningful edge case or failure mode.
- New causal methods, estimands, plotting surfaces, or data loaders should have integration-style coverage where feasible.
- PyMC-heavy tests should use lightweight `sample_kwargs` and deterministic seeds where appropriate.
- User-facing features should include docs, examples, or notebook updates unless the feature is intentionally internal.

## Request Changes When

- The feature is not covered by tests that would fail if the implementation were removed or broken.
- The API shape is unclear, too broad, or inconsistent with existing CausalPy conventions.
- The implementation only works for one backend despite being exposed through model-agnostic code.
- Causal claims, estimands, priors, or interpretation guidance are inaccurate or underspecified.
- The feature creates a public compatibility obligation without enough design clarity.
