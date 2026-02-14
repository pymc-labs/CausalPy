#!/usr/bin/env bash
# Set up the CausalPy development environment for use with Codex, worktrees, or
# any context where conda is not already activated. Creates a conda env at
# ./.conda/env from environment.yml and does an editable install of this repo
# with [dev] extras. Run once per clone/worktree from the repo root.
set -euo pipefail

ENV_PREFIX="./.conda/env"

# Ensure Miniforge conda is discoverable even in clean shells
if ! command -v conda >/dev/null 2>&1; then
  if [ -d "$HOME/miniforge3/condabin" ]; then
    export PATH="$HOME/miniforge3/condabin:$PATH"
  fi
fi

# Fail fast with a helpful message
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Expected Miniforge at \$HOME/miniforge3."
  echo "Fix by installing Miniforge/Miniconda, or adjust PATH in this script."
  exit 1
fi

# Create env if missing
if [ ! -d "$ENV_PREFIX" ]; then
  conda env create -f environment.yml -p "$ENV_PREFIX"
else
  # Optional: keep it in sync (uncomment if you want)
  # conda env update -f environment.yml -p "$ENV_PREFIX" --prune
  :
fi

# Editable install for your normal dev workflow (adjust extras as needed)
conda run -p "$ENV_PREFIX" python -m pip install -e ".[dev]"
