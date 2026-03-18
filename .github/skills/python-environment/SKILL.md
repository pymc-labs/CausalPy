---
name: python-environment
description: Detect and configure a conda-compatible tool, create the CausalPy environment, and run commands inside it. Use before any task that requires Python execution.
---

# Python Environment

Set up and run commands inside the CausalPy conda environment.

## Detect the conda tool

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

## Create the environment

```bash
$CONDA_EXE env create -f environment.yml
```

## Install the package (required after creating or updating the environment)

```bash
$CONDA_EXE run -n CausalPy make setup
```

## Run commands

Always use `run -n` instead of `activate`:

```bash
$CONDA_EXE run -n CausalPy <command>
```

For example: `$CONDA_EXE run -n CausalPy pytest`, `$CONDA_EXE run -n CausalPy pre-commit run --all-files`.

## Update an existing environment

```bash
$CONDA_EXE env update --file environment.yml --prune
```

## Troubleshooting

If you hit issues with an outdated tool, update it:

- **mamba / micromamba**: `$CONDA_EXE self-update`
- **conda**: `conda update -n base conda`

As of 2026-02-13, current versions are conda 26.1.0, mamba/micromamba 2.5.0.
