---
name: parent-child-issues
description: Create a parent GitHub issue with child issues using gh CLI, with permission and capability checks before any write operations.
---

# Parent-child issue workflow (gh CLI)

## 1) Preflight checks (no side effects)

```bash
gh auth status
gh api graphql -f query='query {
  repository(owner:"<owner>", name:"<repo>") {
    viewerPermission
    hasIssuesEnabled
  }
  viewer { login }
}'
```

Proceed only when issues are enabled and permission is sufficient to create/edit issues.

## 2) Verify native sub-issue capability (no side effects)

```bash
gh api graphql -f query='query {
  repository(owner:"<owner>", name:"<repo>") {
    issue(number: 1) {
      number
      parent { number }
      subIssues(first: 1) { nodes { number } }
    }
  }
}'
```

Also confirm mutation availability (at minimum `addSubIssue`):

```bash
gh api graphql -f query='query { __type(name:"Mutation") { fields { name } } }'
```

## 3) Draft issue bodies for user review

Create markdown drafts in `.scratch/issue_summaries/` for:
- one parent tracking issue
- one draft per child issue

Present drafts for user review before posting.

## 4) Create parent and child issues

Create parent:

```bash
gh issue create \
  --repo <owner>/<repo> \
  --title "<parent title>" \
  --body-file "<path-to-parent-body.md>"
```

Create each child:

```bash
gh issue create \
  --repo <owner>/<repo> \
  --title "<child title>" \
  --body-file "<path-to-child-body.md>"
```

## 5) Link child to parent (native hierarchy)

Get GraphQL node IDs:

```bash
gh issue view <parent-number> --repo <owner>/<repo> --json id,number,title
gh issue view <child-number> --repo <owner>/<repo> --json id,number,title
```

Attach child to parent:

```bash
gh api graphql \
  -f query='mutation($issue: ID!, $sub: ID!) {
    addSubIssue(input: {issueId: $issue, subIssueId: $sub}) {
      issue { number title url }
      subIssue { number title url }
    }
  }' \
  -f issue='<PARENT_NODE_ID>' \
  -f sub='<CHILD_NODE_ID>'
```

## 6) Discover existing sub-issues

When evaluating an issue that may already have sub-issues, always use the `subIssues` GraphQL field. Do **not** use `trackedIssues` / `trackedInIssues` -- those only cover older markdown task-list tracking and will miss native sub-issues.

```bash
gh api graphql -f query='query {
  repository(owner:"<owner>", name:"<repo>") {
    issue(number: <issue_number>) {
      subIssues(first: 50) {
        nodes { number title state url }
      }
      parent { number title state url }
    }
  }
}'
```

## 7) Verify links rendered correctly

```bash
gh api graphql -f query='query {
  repository(owner:"<owner>", name:"<repo>") {
    issue(number: <parent-number>) {
      number
      title
      subIssues(first: 50) { nodes { number title url } }
    }
  }
}'
```

## Fallback if native sub-issues are unavailable

Do **not** apply fallback automatically.

When native sub-issues are unavailable, first inform the user:
- what capability check failed (missing `parent`/`subIssues` fields or no `addSubIssue` mutation)
- that native hierarchy cannot be created in this repo/account context
- that a markdown tracking-issue fallback is available

Then explicitly ask whether they want to proceed with fallback. Only continue after user confirmation.

Use a tracking issue body with markdown task links:

```md
- [ ] #123 Child issue one
- [ ] #124 Child issue two
```

This renders correctly on GitHub and preserves parent-child tracking even without native hierarchy.
