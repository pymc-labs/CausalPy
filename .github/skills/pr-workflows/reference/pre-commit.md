---
name: pre-commit
description: Run pre-commit checks and handle auto-fix output.
---

# Pre-commit Usage

## Run checks
```bash
pre-commit run --all-files
```

## If hooks modify files
1. Re-stage modified files.
2. Re-run pre-commit if needed.
3. Commit after the working tree is clean.

## Common fixes
- Formatting or lint auto-fixes
- Regenerated docs/assets
