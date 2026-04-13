# Notebook Runner

This script runs Jupyter notebooks from `docs/source/notebooks/` to validate they execute without errors.

## How It Works

1. **Mocks `pm.sample()`** — Replaces MCMC sampling with prior predictive (1 chain × 100 draws) for speed
2. **Uses Papermill** — Executes notebooks programmatically
3. **Clears saved outputs** — Avoids widget state issues during execution
4. **Guards widget updates** — Patches nbclient to ignore display_id assertion errors
5. **Discards outputs** — Only checks for errors, doesn't save results

## Dependencies

The notebook runner mirrors the CI setup and expects a full docs/test environment.

1. **Install Python dependencies**

   ```bash
   pip install -e ".[test,docs]"
   ```

   This brings in Papermill, Jupyter, nbclient, and notebook-related dependencies.

2. **Install Graphviz (system dependency)**

   - macOS:
     ```bash
     brew install graphviz
     ```
   - Ubuntu/Debian:
     ```bash
     sudo apt-get update && sudo apt-get install -y graphviz
     ```

3. **Optional: parallel execution**

   ```bash
   pip install joblib
   ```

## Notes

- The runner executes using the `python3` Jupyter kernel. Ensure your environment
  provides that kernel (e.g., from `ipykernel` installed via the docs extras).
- The CI workflow uses Python 3.12 and installs the same extras.

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

# Run notebooks in parallel (requires joblib)
python scripts/run_notebooks/runner.py --parallel
```

## CI Integration

The GitHub Actions workflow (`.github/workflows/test_notebook.yml`) runs this script in parallel:
- Job 1: PyMC notebooks
- Job 2: Sklearn notebooks
- Job 3: Other notebooks

## Files

- `runner.py` — Main script
- `injected.py` — Code injected into notebooks to mock `pm.sample()`
- `skip_notebooks.yml` — List of notebooks to skip (incompatible with mock sampling)
