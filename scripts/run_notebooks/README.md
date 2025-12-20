# Notebook Runner

This script runs Jupyter notebooks from `docs/source/notebooks/` to validate they execute without errors.

## How It Works

1. **Mocks `pm.sample()`** — Replaces MCMC sampling with prior predictive (10 draws) for speed
2. **Uses Papermill** — Executes notebooks programmatically
3. **Discards outputs** — Only checks for errors, doesn't save results

## Usage

```bash
# Run all notebooks
python scripts/run_notebooks/runner.py

# Run only PyMC notebooks
python scripts/run_notebooks/runner.py --pattern "*_pymc*.ipynb"

# Run only sklearn notebooks
python scripts/run_notebooks/runner.py --pattern "*_skl*.ipynb"

# Exclude PyMC and sklearn notebooks (run others)
python scripts/run_notebooks/runner.py --exclude-pattern _pymc --exclude-pattern _skl
```

## CI Integration

The GitHub Actions workflow (`.github/workflows/test_notebook.yml`) runs this script in parallel:
- Job 1: PyMC notebooks
- Job 2: Sklearn notebooks
- Job 3: Other notebooks

## Files

- `runner.py` — Main script
- `injected.py` — Code injected into notebooks to mock `pm.sample()`
