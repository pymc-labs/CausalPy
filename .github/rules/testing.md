---
globs: causalpy/tests/**
---

## Testing preferences

- Write all Python tests as `pytest` style functions, not unittest classes.
- Use descriptive function names starting with `test_`.
- Prefer fixtures over setup/teardown methods.
- Use assert statements directly, not self.assertEqual.

## Testing approach

- Never create throwaway test scripts or ad hoc verification files.
- All tests go in the `causalpy/tests/` directory following the project structure.
- Tests should be runnable with the rest of the suite (`python -m pytest`).
- Even for quick verification, write it as a real test that provides ongoing value.
- Preference should be given to integration tests, but unit tests are acceptable for core functionality to maintain high code coverage.
- Tests should remain quick to run. Tests involving MCMC sampling with PyMC should use custom `sample_kwargs` to minimize the computational load.

## Sandbox and permissions

- **PyMC/PyTensor require filesystem access** outside the workspace (e.g. `~/.pytensor/compiledir_*` for C compilation cache, `~/.matplotlib` for font cache).
- **Always use `required_permissions: ["all"]`** when running `pytest`, `make doctest`, or any command that imports PyMC, PyTensor, or matplotlib.
- **If a test run shows `compiledir ... you don't have read, write or listing permissions`**, that is a sandbox problem, not a code problem.
