# Issue to Pull Request

Transform a GitHub issue into a complete pull request through a structured, interactive workflow.

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

## Workflow

### Phase 1: Issue Discovery

1. **Fetch the issue:**
   ```bash
   gh issue view <issue_number> --json title,body,labels,comments,state
   ```

2. **Build understanding:**
   - Parse the issue title, body, and all comments
   - Identify the problem statement, acceptance criteria, and any edge cases
   - Note any related files, modules, or tests mentioned
   - Check for linked issues or PRs

3. **Estimate complexity:**
   - **Small**: Single file change, clear solution, isolated scope
   - **Medium**: Multiple files, requires understanding context, some testing
   - **Large**: Architectural changes, multiple components, extensive testing
   - **Requires Planning**: If Large, inform user that a detailed plan should be created first using `/make_plan` or the research workflow

4. **Present understanding to user:**
   > "Based on issue #<number>: '<title>'
   >
   > **My understanding:** <summary of what needs to be done>
   >
   > **Affected areas:** <list of files/modules likely to change>
   >
   > **Estimated complexity:** <Small/Medium/Large>
   >
   > **Approach:** <brief technical approach>
   >
   > Is this understanding correct? Any adjustments before I proceed?"

   **Wait for user confirmation before continuing.**

### Phase 2: Implementation

1. **Create a feature branch:**
   ```bash
   git checkout -b issue-<number>-<short-description>
   ```

2. **Implement the solution:**
   - Follow the project's code structure and style guidelines
   - Refer to `AGENTS.md` for coding conventions
   - Write tests for new functionality in `causalpy/tests/`
   - Update documentation if needed

3. **Validation loop:**
   Iterate until all checks pass:

   a. **Run pre-commit:**
      ```bash
      pre-commit run --all-files
      ```
      - Fix any linting, formatting, or type errors
      - Re-run until clean

   b. **Run tests:**
      ```bash
      python -m pytest causalpy/tests/ -x -v
      ```
      - If tests fail, diagnose and fix
      - Add new tests if coverage is insufficient

   c. **If stuck after 3 iterations:**
      > "I've encountered an issue I need help with:
      >
      > **Problem:** <describe the issue>
      >
      > **What I've tried:** <list attempts>
      >
      > **Options I see:** <possible solutions>
      >
      > How would you like to proceed?"

      **Wait for user guidance.**

### Phase 3: Commit and PR Preparation

1. **Stage and commit changes:**
   - Use atomic, focused commits
   - Write clear commit messages in imperative mood
   - Reference the issue number: `Fixes #<number>` or `Addresses #<number>`

2. **Generate PR markdown file:**
   Create `.github/pr_summaries/<issue_number> - <short-description>.md` with:

   ```markdown
   ## Summary

   <Brief description of what this PR does>

   Fixes #<issue_number>

   ## Changes

   - <List of key changes>
   - <Bullet points for each significant modification>

   ## Testing

   - <How the changes were tested>
   - <Any new tests added>

   ## Checklist

   - [ ] Pre-commit checks pass
   - [ ] All tests pass
   - [ ] Documentation updated (if applicable)
   - [ ] Follows project coding conventions
   ```

3. **Present PR draft to user:**
   > "I've prepared the following for PR creation:
   >
   > **Branch:** `issue-<number>-<description>`
   >
   > **Commits:** <list of commits>
   >
   > **PR Description:** <show the markdown content>
   >
   > Ready to create the pull request? (yes/no)"

   **Wait for user confirmation.**

### Phase 4: Create Pull Request

1. **Push the branch:**
   ```bash
   git push -u origin issue-<number>-<short-description>
   ```

2. **Create the PR:**
   ```bash
   gh pr create --title "<PR title from issue>" --body-file .github/pr_summaries/<issue_number>\ -\ <short-description>.md --base main
   ```

3. **Report success:**
   > "✅ Pull request created successfully!
   >
   > **PR URL:** <link>
   >
   > The PR markdown file has been saved to `.github/pr_summaries/<filename>.md` for reference."

4. **Cleanup (optional):**
   - Offer to delete the PR markdown file if not needed
   - Suggest next steps (review, CI checks, etc.)

## Important Notes

- **Always wait for user confirmation** at key decision points (after understanding, before PR creation)
- **Never force push** or modify git history without explicit permission
- **Be transparent** about what commands you're running and why
- **Commits should be authored solely by the user** - no "Generated with Cursor" messages
- **If the issue is unclear**, ask for clarification rather than making assumptions
- **For large issues**, recommend breaking into smaller PRs or creating a detailed plan first

## Error Handling

- **Issue not found:** Verify the issue number and repository
- **Permission denied:** Check `gh auth status` and repository access
- **Merge conflicts:** Alert the user and provide options (rebase, merge main, etc.)
- **CI failures:** Investigate and fix, or report to user if external issue
