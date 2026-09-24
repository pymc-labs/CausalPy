# Notebook Runner

This script runs Jupyter notebooks from the gallery (`docs/source/notebooks/`) and knowledgebase (`docs/source/knowledgebase/`) to validate they execute without errors.

## How It Works

1. **Mocks `pm.sample()`** — Replaces MCMC sampling with prior predictive (1 chain × 100 draws) for speed
2. **Uses Papermill** — Executes notebooks programmatically
3. **Runs serially** — Executes one notebook at a time to control memory use
4. **Clears saved outputs** — Avoids widget state issues during execution
5. **Guards widget updates** — Patches nbclient to ignore display_id assertion errors
6. **Discards outputs** — Only checks for errors, doesn't save results

## Dependencies

The notebook runner mirrors the CI setup and expects a full docs/test environment.

1. **Install Python dependencies**

   Default (uv):

   ```bash
   uv sync --locked --extra test --extra docs
   ```

   Note: CI's notebook workflow additionally installs the docs-only PyMC-Marketing
   transition snapshot from `docs/requirements.txt` first via `uv pip install`
   (unlocked — its pins conflict with `uv.lock`'s floors). See
   `.github/workflows/test_notebook.yml`.

   Alternative (conda/micromamba environment, or plain pip):

   ```bash
   pip install -e ".[test,docs]"
   ```

   Either way, this brings in Papermill, Jupyter, nbclient, and notebook-related dependencies.

2. **Install Graphviz (system dependency)**

   - macOS:

     ```bash
     brew install graphviz
     ```

   - Ubuntu/Debian:

     ```bash
     sudo apt-get update && sudo apt-get install -y graphviz
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
python scripts/run_notebooks/runner.py --pattern "*-pymc.ipynb"

# Run only sklearn notebooks
python scripts/run_notebooks/runner.py --pattern "*-sklearn.ipynb"

# Exclude PyMC and sklearn notebooks (run others)
python scripts/run_notebooks/runner.py --exclude-pattern pymc --exclude-pattern sklearn

# Run the knowledgebase collection
python scripts/run_notebooks/runner.py --collection knowledgebase

# List an exact pair without executing or changing files
python scripts/run_notebooks/runner.py --list \
  --notebook docs/source/notebooks/ancova-pymc.ipynb \
  --notebook docs/source/knowledgebase/custom_pymc_models.ipynb

# Refresh one notebook in place without mock injection
python scripts/run_notebooks/runner.py --full \
  --notebook docs/source/notebooks/ancova-pymc.ipynb
```

The notebooks listed in `skip_notebooks.yml` are incompatible with mock injection and are omitted from collection/pattern mock runs. Selecting one explicitly in mock mode fails with a clear error; select it with `--full --notebook ...` for controlled, serialized execution in an environment containing its optional dependencies.

## CI Integration

The GitHub Actions workflow (`.github/workflows/test_notebook.yml`) runs four serial matrix entries (`max-parallel: 1`). Superseded runs are canceled so only the newest commit occupies the serial queue:

- Job 1: PyMC notebooks
- Job 2: Sklearn notebooks
- Job 3: Other notebooks
- Job 4: Knowledgebase notebooks

## Files

- `runner.py` — Main script
- `injected.py` — Code injected into notebooks to mock `pm.sample()`
- `skip_notebooks.yml` — List of notebooks to skip (incompatible with mock sampling)
