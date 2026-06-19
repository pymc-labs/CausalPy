# PR: Fix numpydoc docstring section ordering

Closes #970

## Issue Summary

The pending `numpydoc` update in PR #969 surfaced stricter validation for NumPy docstring section order. Several public experiment and PyMC model docstrings had `Examples` before `Notes` or `References`, or used a singular `Example` heading.

## Root Cause

Older `numpydoc` validation accepted section ordering and heading variants that `v1.11.0rc0` now flags under `GL07`.

## Solution

Reordered affected docstring sections to the NumPy convention and normalized singular `Example` headings to `Examples`.

## Changes Made

- `causalpy/experiments/panel_regression.py`: Moved `Notes` before `Examples`.
- `causalpy/experiments/instrumental_variable.py`: Renamed `Example` to `Examples`.
- `causalpy/experiments/staggered_did.py`: Moved `References` before `Examples` and normalized the heading.
- `causalpy/experiments/synthetic_control.py`: Moved `Notes` before `Examples`.
- `causalpy/pymc_models.py`: Normalized singular `Example` headings to `Examples`.
- `causalpy/experiments/inverse_propensity_weighting.py`: Renamed `Example` to `Examples`.
- `causalpy/experiments/synthetic_difference_in_differences.py`: Moved `Notes` and `References` before `Examples`.

## Testing

- [x] Existing tests pass
- [x] New tests added (not applicable; docstring-only validation fix)
- [x] Manual verification completed

## Notes

The local pre-commit hook is still pinned to `numpydoc v1.10.0`, but these edits address the stricter section ordering described in #970.
