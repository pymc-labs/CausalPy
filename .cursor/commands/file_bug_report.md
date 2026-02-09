# File Bug Report

Transform a discovered bug into a well-documented GitHub issue.

## Prerequisites Check

Before starting, verify the GitHub CLI is available and authenticated:

1. **Check `gh` is installed:**
   ```bash
   gh --version
   ```
   - If not found, guide the user to install it:
     - macOS: `brew install gh`
     - Linux: See https://github.com/cli/cli/blob/trunk/docs/install_linux.md
     - Windows: `winget install --id GitHub.cli`

2. **Check authentication:**
   ```bash
   gh auth status
   ```
   - If not authenticated, guide: `gh auth login`

3. **Sandbox considerations:**
   - GitHub CLI commands require access to credentials stored outside the workspace
   - **Always use `required_permissions: ["all"]`** for any `gh` commands

## Workflow

### Phase 1: Bug Investigation

1. **Gather initial information from the user:**
   - Error message or unexpected behavior
   - Steps to reproduce (if known)
   - Relevant code or notebook

2. **Investigate the bug:**
   - Locate the relevant source code
   - Reproduce the error if possible (write a minimal test case)
   - Identify the root cause
   - Check if this is a known issue:
     ```bash
     gh issue list --state open --search "<keywords>"
     ```

3. **If a solution is apparent:**
   - Note the proposed fix in the issue body
   - This helps future contributors (including yourself) address it quickly

### Phase 2: Draft Issue

1. **Create the issue markdown file:**
   Save to `.scratch/issue_summaries/<short-description>.md` with this structure:

   ```markdown
   ## Description

   <Clear, concise description of the bug>

   ## Steps to Reproduce

   1. <Step 1>
   2. <Step 2>
   3. ...

   ## Expected Behavior

   <What should happen>

   ## Actual Behavior

   <What actually happens, including error messages>

   ## Environment

   - CausalPy version: <version or commit>
   - Python version: <version>
   - OS: <operating system>

   ## Proposed Solution (if known)

   <Brief description of the fix, or remove this section if unknown>

   ## Additional Context

   <Any other relevant information, screenshots, or code snippets>
   ```

   **Formatting note:** Do not hard-wrap lines in markdown drafts; keep paragraphs on a single line and rely on editor auto-wrapping.

2. **Present the draft to the user:**
   > "I've drafted the following bug report:
   >
   > **Title:** `<proposed title>`
   >
   > **Body:**
   > <show the markdown content>
   >
   > Would you like to make any changes before filing?"

   **Wait for user confirmation.**

### Phase 3: File the Issue

1. **Create the issue:**
   ```bash
   gh issue create --title "<descriptive title>" --body-file .scratch/issue_summaries/<short-description>.md --label "bug"
   ```

2. **Report success:**
   > "✅ Bug report filed successfully!
   >
   > **Issue URL:** <link>
   >
   > **Issue Number:** #<number>"

3. **Cleanup:**
   Delete the markdown file (it's gitignored and no longer needed):
   ```bash
   rm .scratch/issue_summaries/<short-description>.md
   ```

## Guidelines

- **Keep titles concise but descriptive:** Bad: "Error in code". Good: "KeyError when calling result.summary() with missing covariate"
- **Include reproducible examples (if possible):** Minimal code that triggers the bug
- **Don't speculate excessively:** If unsure about the cause, say so
- **One bug per issue:** If multiple bugs are discovered, file separate issues

## Error Handling

- **Duplicate issue found:** Link to the existing issue instead of creating a new one
- **Cannot reproduce:** File anyway with available information, note that reproduction failed
- **Permission denied:** Check `gh auth status` and repository access (sandbox permissions covered in Prerequisites)
