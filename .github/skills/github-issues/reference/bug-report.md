---
name: bug-report
description: Transform a discovered bug into a well-documented GitHub issue.
---

# Bug Report Workflow

## Gather information
- Error message or unexpected behavior
- Steps to reproduce (if known)
- Relevant code or notebook

## Investigate
- Locate relevant source code
- Reproduce with a minimal example if possible
- Check for duplicates:
  ```bash
  gh issue list --state open --search "<keywords>"
  ```

## Draft issue
Create `.scratch/issue_summaries/<short-description>.md` with the bug-specific template below, then follow the general issue creation steps in `issue-creation.md` for review, creation, labeling, and cleanup.

Bug-specific template:
```markdown
## Description
<Clear, concise description of the bug>

## Steps to Reproduce
1. <Step 1>
2. <Step 2>

## Expected Behavior
<What should happen>

## Actual Behavior
<What actually happens>

## Environment
- CausalPy version:
- Python version:
- OS:

## Proposed Solution (if known)
<Brief fix idea>

## Additional Context
<Any other relevant info>
```

## Next steps
- Present the draft for review.
- Use the `issue-creation` workflow to file and label the issue.
