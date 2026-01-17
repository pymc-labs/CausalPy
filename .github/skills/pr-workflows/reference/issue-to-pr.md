---
name: issue-to-pr
description: Transform a GitHub issue into a complete pull request through a structured workflow.
---

# Issue → PR Workflow

## Discovery
1. Fetch issue details:
   ```bash
   gh issue view <issue_number> --json title,body,labels,comments,state
   ```
2. Summarize the problem, acceptance criteria, and affected files.
3. Estimate complexity and share approach before implementation.

## Implementation
1. Create a branch: `git checkout -b issue-<number>-<short-description>`
2. Implement the fix, following `AGENTS.md` policies.
3. Run pre-commit and tests until green.

## Prepare PR
1. Create `.github/pr_summaries/<issue_number> - <short-description>.md` with:
   - Summary
   - Fixes #<issue_number>
   - Changes
   - Testing
   - Checklist
2. Share the PR draft with the user before creating the PR.

## Create PR
```bash
git push -u origin issue-<number>-<short-description>
gh pr create --title "<PR title>" --body-file .github/pr_summaries/<issue_number> - <short-description>.md --base main
```
