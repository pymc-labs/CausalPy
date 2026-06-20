---
name: prek
description: Run prek checks and handle auto-fix output.
---

# Prek Usage

## Run checks

```bash
# Fast iteration on changed files
prek run --files path/to/file.py

# Full verification before push/commit
prek run --all-files

# Python source/test changes: local patch coverage gate
make test-patch-cov

# Local checks before push (CI is the safety net)
prek run --all-files
make test-patch-cov                    # when causalpy/ source or tests changed
make doctest                           # when causalpy/ docstring examples may break
python scripts/run_notebooks/runner.py --pattern 'foo.ipynb'  # API/notebook changes (CI-style mock)
make test                              # large cross-cutting PRs

# Slow optional: refresh committed notebook outputs (NOT what CI runs)
# make run_notebooks_full
```

See [AGENTS.md](../../../AGENTS.md) § Before push for details.

## If hooks modify files

1. Re-stage modified files.
2. Re-run prek if needed.
3. Commit after the working tree is clean.

## Common fixes

- Formatting or lint auto-fixes
- Regenerated docs/assets
