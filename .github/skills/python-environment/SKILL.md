---
name: python-environment
description: Detect and configure a conda-compatible tool, reuse an existing CausalPy environment when possible, create it only when needed, and run commands inside it. Use before tasks that need the project environment, such as importing project code, running tests, building docs, or invoking repo tooling.
---

# Python Environment

Set up and run commands inside the CausalPy conda environment.

## Decide whether the env is required

Use the `CausalPy` env when the command:

- imports project code (for example `import causalpy` or project modules)
- runs tests
- builds docs
- invokes repo tooling such as `make`, `prek`, or notebook execution

For simple inspection helpers that only read local text/JSON or use the Python standard library, any Python on `PATH` is acceptable.

## Reuse before creating

Do the least work that will get the task done:

1. Reuse an existing `CausalPy` env if one is already available.
2. If `run -n CausalPy` cannot resolve the env, check whether it exists under a different prefix and use `run -p`.
3. Only create the env if no suitable existing env is available.
4. Only update the env or rerun `make setup` when dependencies changed, the editable install is stale, or the current checkout has not been installed into that env yet.

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

## Create the environment only if needed

If no suitable existing env can be reused, create it:

```bash
$CONDA_EXE env create -f environment.yml
```

## Install the package only when needed

Run `make setup` after creating or updating the env. Also rerun it when using a different git worktree if that env has not been installed against the current checkout yet.

```bash
$CONDA_EXE run -n CausalPy make setup
```

## Run commands

Prefer `run -n` instead of `activate`:

```bash
$CONDA_EXE run -n CausalPy <command>
```

For example: `$CONDA_EXE run -n CausalPy pytest`, `$CONDA_EXE run -n CausalPy prek run --all-files`.

## Update an existing environment

```bash
$CONDA_EXE env update --file environment.yml --prune
```

## Troubleshooting

### Named env cannot be resolved

If `$CONDA_EXE run -n CausalPy ...` fails with errors such as `The given prefix does not exist`:

```bash
$CONDA_EXE env list
$CONDA_EXE run -p "/full/path/to/CausalPy" <command>
```

Keep using `run -p` with that full prefix for the rest of the session.

Note: a shell `conda` function may still resolve through the same libmamba installation as `mamba`, so switching from `mamba` to `conda` does not necessarily change which env store is searched.

### Git worktrees and remote machines

Git worktrees do not require a fresh env per agent session. Prefer reusing an existing env to save time. The main caveat is that this repo uses editable installs, so one shared env can point at whichever checkout most recently ran `make setup`.

- For ordinary local work on one checkout, reuse the existing env.
- For long-lived parallel worktrees, one env per worktree is the safest option, but do not create one unless needed.
- On a fresh remote machine or ephemeral container, create the env once. On a persistent remote machine with an existing env, reuse it.

If you hit issues with an outdated tool, update it:

- **mamba / micromamba**: `$CONDA_EXE self-update`
- **conda**: `conda update -n base conda`

As of 2026-02-13, current versions are conda 26.1.0, mamba/micromamba 2.5.0.
