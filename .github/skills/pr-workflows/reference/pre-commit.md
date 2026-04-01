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
```

## If hooks modify files
1. Re-stage modified files.
2. Re-run prek if needed.
3. Commit after the working tree is clean.

## Common fixes
- Formatting or lint auto-fixes
- Regenerated docs/assets
