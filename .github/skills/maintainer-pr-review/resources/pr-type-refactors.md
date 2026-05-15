# Refactor PR Review

Use this resource when a PR reorganizes code, renames internals, extracts helpers, changes module boundaries, reduces duplication, or rewrites implementation without intending user-visible behavior changes.

## Review Focus

- Establish whether behavior is intended to be unchanged. If behavior changes are present, classify the PR as mixed and also read the relevant feature or bug-fix resource.
- Compare public signatures, documented behavior, exceptions, returned objects, plots, and data shapes against the base branch.
- Check that extracted abstractions remove real complexity and follow existing project boundaries rather than creating generic helpers prematurely.
- Look for subtle behavior changes from pandas index handling, xarray dimensions, formula evaluation, random seeds, dtype coercion, plotting defaults, or exception types.
- Confirm imports remain stable and do not introduce circular dependencies, heavier import-time costs, or optional dependency leaks.
- Review deletions carefully: removed branches, tests, docs, or compatibility handling may encode shipped behavior.

## Required Evidence

- Existing tests that cover the touched behavior should still pass, or the review should note that they were not run.
- Meaningful refactors should usually include unchanged or strengthened tests around the moved behavior.
- Broad refactors should have a clear rationale in the PR description and should avoid unrelated formatting churn.

## Review Output Emphasis

When writing the final review, foreground whether behavior is meant to remain unchanged, any accidental semantic change, public API or import compatibility, preservation evidence from tests, and whether mechanical movement should be split from behavior changes.

## Request Changes When

- The refactor changes behavior without acknowledging it.
- The new abstraction obscures causal/statistical meaning or makes backend-specific behavior harder to inspect.
- Tests were weakened, removed, or made less representative to accommodate the refactor.
- The diff is too broad to review confidently without splitting mechanical movement from semantic changes.
- Public imports, docs, or examples break because internal names moved.
