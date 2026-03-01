---
name: issue-to-pr
description: Transform a GitHub issue into a complete pull request through a structured workflow.
---

# Issue → PR Workflow

## Branch preflight (required before any code edits)
1. Check current branch and working tree state.
2. Switch to `main` and update it from remote.
3. Create the issue branch from `main`: `git checkout -b issue-<issue_number>-<short-description>`
4. Only start implementation after confirming the new branch is based on `main`.

## Discovery
1. Fetch issue details:
   ```bash
   gh issue view <issue_number> --json title,body,labels,comments,state
   ```
2. Summarize the problem, acceptance criteria, and affected files.
3. Estimate complexity and share approach before implementation.

## Implementation
1. Implement the fix, following `AGENTS.md` policies.
2. Run pre-commit and tests until green (see `pre-commit.md`).

## Prepare PR
1. Create `.scratch/pr_summaries/<issue_number> - <short-description>.md` with:
   - Summary
   - Fixes #<issue_number>
   - Changes
   - Testing
   - Checklist
2. Share the PR draft with the user before creating the PR.
3. Commit changes following `committing.md`.

## Create PR
```bash
git push -u origin issue-<issue_number>-<short-description>
gh pr create --title "<PR title>" --body-file .scratch/pr_summaries/<issue_number> - <short-description>.md --base main
```
