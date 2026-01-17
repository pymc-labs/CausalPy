---
name: issue-creation
description: Create a GitHub issue with a reviewed markdown body and appropriate labels.
---

# Issue Creation Workflow

## Prerequisites
1. Verify GitHub CLI is installed: `gh --version`
2. Check authentication: `gh auth status`

## Drafting the issue
1. Create a markdown file (e.g., `issue.md`) with:
   - Problem statement or feature request
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior (for bugs)
   - Relevant code or error output
   - Proposed solution (if known)
2. Present the draft to the user for review and edits.

## Create the issue
After approval, run:
```bash
gh issue create --title "<descriptive title>" --body-file issue.md
```

Add labels with `--label`, e.g.:
```bash
gh issue create --title "<descriptive title>" --body-file issue.md --label "bug"
```

## Cleanup
Delete the temporary `issue.md` after submission.
