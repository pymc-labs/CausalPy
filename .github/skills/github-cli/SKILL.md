---
name: github-cli
description: Interact with GitHub issues, PRs, releases via the gh CLI. Use when reading or creating issues, viewing pull requests, checking CI status, or any GitHub interaction.
---

# GitHub CLI

The preferred source of truth for GitHub is the GitHub CLI (`gh`), not web fetch.

## Confirm Tooling

Before working with GitHub, verify `gh` is available and authenticated:

```bash
command -v gh                                           # Check gh is installed
gh auth status                                          # Check authentication
gh repo view --json nameWithOwner -q .nameWithOwner     # Verify repo access
```

If `gh` is unavailable, install it:
- macOS: `brew install gh`
- Linux: See https://github.com/cli/cli/blob/trunk/docs/install_linux.md
- Windows: `winget install --id GitHub.cli`

Then authenticate: `gh auth login`

## When to Fall Back to Web Fetch

Only fetch github.com pages directly if:
- `gh` is unavailable, OR
- `gh auth status` fails / repo permissions block access

Prefer `gh` commands or `gh api` over copying HTML from the browser.

## Issues

```bash
gh issue view <num> --comments    # View issue with all comments
gh issue list                     # List open issues
gh issue list --state all         # List all issues
```

**Creating issues:**

1. Create a markdown file (e.g., `issue.md`) with problem statement, reproduction steps, expected vs actual behavior, and proposed solution
2. Present draft to user for review before filing
3. Create: `gh issue create --title "<title>" --body-file issue.md --label "bug"`
4. Clean up: Delete temporary `issue.md` file

Discover available labels: `gh label list --limit 100`

## Pull Requests

```bash
gh pr view <num>                  # View PR details
gh pr view <num> --comments       # View PR with comments
gh pr list                        # List open PRs
gh pr checks <num>                # View CI status
gh pr diff <num>                  # View PR diff
```

## Other Resources

```bash
gh release list                   # List releases
gh run list                       # List workflow runs
gh api <endpoint>                 # Direct API access for anything else
```
