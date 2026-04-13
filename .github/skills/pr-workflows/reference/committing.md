---
name: committing
description: Create clean, focused commits with user confirmation and prek checks.
---

# Committing Changes

## Process
1. Review changes:
   - `git status`
   - `git diff`
2. Group files into logical commits.
3. Draft clear, imperative commit messages focused on the why.
4. Ask for user confirmation before committing.

## Commit steps
1. Run prek (see `pre-commit.md` for details). Prefer scoped runs while iterating, then `prek run --all-files` before final commit.
2. Stage only relevant files (avoid `git add -A` / `git add .`).
3. Create the commit with a clear message.

## Notes
- Commits should be authored by the user (no co-author lines).
- If hooks modify files, re-add and retry the commit.
