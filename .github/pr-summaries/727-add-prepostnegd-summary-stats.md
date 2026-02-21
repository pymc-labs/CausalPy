# PR: Add group-level outputs to PrePostNEGD summary

Closes #727

## Issue Summary

`PrePostNEGD.summary()` only reported formula, causal impact, and model coefficients. The issue requested experiment-specific outputs to make the summary more informative.

## Root Cause

The summary method had no implementation for experiment-specific diagnostics beyond the generic treatment-effect and coefficient output.

## Solution

Added a group-level descriptive statistics section to the `PrePostNEGD` summary output, including sample size and pre/post means by group, and added a regression test to verify those fields are printed.

## Changes Made

- `causalpy/experiments/prepostnegd.py`: added `_group_level_summary_stats()` and updated `summary()` to print group-level `n`, `pre_mean`, and `post_mean`.
- `causalpy/tests/test_integration_pymc_examples.py`: updated `test_ancova` to capture summary output and assert new experiment-specific lines are present.

## Testing

- [x] Existing tests pass
- [x] New tests added (if applicable)
- [x] Manual verification completed

## Notes

The documentation notebook that displays this summary output and should be re-run is `docs/source/notebooks/ancova_pymc.ipynb`.
