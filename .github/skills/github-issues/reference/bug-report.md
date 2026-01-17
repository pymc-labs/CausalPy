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
Create `.github/issue_summaries/<short-description>.md` with:
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

Present the draft to the user before posting.

## File the issue
```bash
gh issue create --title "<descriptive title>" --body-file .github/issue_summaries/<short-description>.md --label "bug"
```

## Cleanup
Remove the markdown file after filing.
