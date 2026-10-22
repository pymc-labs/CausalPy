---
name: python-environment
description: Detect, configure, and use the project's Python environment (uv by default, conda-compatible tool as a fallback). Use before tasks that need the project environment, such as importing project code, running tests, building docs, or invoking repo tooling.
---

# Python Environment

Set up and run commands inside the CausalPy dev environment. **uv is the default; a conda-compatible tool is the fallback** when `uv` is not available.

## Decide whether the env is required

Use the project environment when the command:

- imports project code (for example `import causalpy` or project modules)
- runs tests
- builds docs
- invokes repo tooling such as `make`, `prek`, or notebook execution

For simple inspection helpers that only read local text/JSON or use the Python standard library, any Python on `PATH` is acceptable.

## Default: uv

### Reuse before creating

Do the least work that will get the task done:

1. Reuse an existing `.venv` (created by `uv sync`) if the checkout already has one.
2. Only run `uv sync` again when dependencies changed, the editable install is stale, or the current checkout has not been synced yet.

### Set up the environment

```bash
uv sync --locked --extra dev --extra docs --extra test --extra lint
uv run prek install -f
```

`--locked` fails fast if `uv.lock` has drifted from `pyproject.toml` instead of silently re-resolving.

### Run commands

There is no environment to activate — prefix every command with `uv run`:

```bash
uv run pytest
uv run make test
uv run prek run --all-files
```

### Update the environment

Re-run `uv sync --locked ...` with the same extras after pulling changes that touch `pyproject.toml` or `uv.lock`.

### Git worktrees and remote machines

uv does not require a fresh `.venv` per agent session, but because this repo uses editable installs, one shared `.venv` points at whichever checkout most recently ran `uv sync`.

- For ordinary local work on one checkout, reuse the existing `.venv`.
- For long-lived parallel worktrees, one `.venv` per worktree is the safest option (`uv sync` inside each), but do not create one unless needed.
- On a fresh remote machine or ephemeral container, run `uv sync` once. On a persistent remote machine with an existing `.venv`, reuse it.

## Fallback: conda-compatible tool

Use this path only when `uv` is not available, or when the task specifically requires the conda/micromamba alternative (e.g. validating `environment.yml`).

### Detect the conda tool

Use whichever of `mamba`, `micromamba`, or `conda` is available (checked in that order):

```bash
# Check for mamba, micromamba, or conda (in preference order) on $PATH
CONDA_EXE=$(for c in mamba micromamba conda; do command -v "$c" &>/dev/null && echo "$c" && break; done)
```

If `CONDA_EXE` is empty, no conda-compatible tool was found. Propose installing micromamba to the user:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

After installation, set `CONDA_EXE=micromamba`.

### Create the environment only if needed

If no suitable existing env can be reused, create it:

```bash
$CONDA_EXE env create -f environment.yml
```

### Install the package only when needed

Run `make setup-conda` after creating or updating the env, from inside an active/running conda env. Also rerun it when using a different git worktree if that env has not been installed against the current checkout yet.

```bash
$CONDA_EXE run -n CausalPy make setup-conda
```

### Run commands

Never use `$CONDA_EXE activate`, instead use `$CONDA_EXE run -n CausalPy <command>`.

```bash
$CONDA_EXE run -n CausalPy <command>
```

For example: `$CONDA_EXE run -n CausalPy pytest`, `$CONDA_EXE run -n CausalPy prek run --all-files`.

### Update an existing environment

```bash
$CONDA_EXE env update --file environment.yml --prune
```

### Troubleshooting

#### Named env cannot be resolved

If `$CONDA_EXE run -n CausalPy ...` fails with errors such as `The given prefix does not exist`:

```bash
$CONDA_EXE env list
$CONDA_EXE run -p "/full/path/to/CausalPy" <command>
```

Keep using `run -p` with that full prefix for the rest of the session.

#### Updating an outdated conda tool

If you hit issues with an outdated tool, update it:

- **mamba / micromamba**: `$CONDA_EXE self-update`
- **conda**: `conda update -n base conda`

As of 2026-02-13, current versions are conda 26.1.0, mamba/micromamba 2.5.0.
