---
name: issue-evaluation
description: Analyze an issue, check current relevance, and propose next steps or a resolution path.
---

# Issue Evaluation Workflow

## Fetch issue details
```bash
gh issue view <issue_number> --json number,title,body,state,labels,comments,assignees
```

## Check for sub-issues (native hierarchy)

Always check whether an issue has native sub-issues. The `trackedIssues` / `trackedInIssues` GraphQL fields only cover older markdown task-list tracking and will miss native sub-issues.

```bash
gh api graphql -f query='query {
  repository(owner:"<owner>", name:"<repo>") {
    issue(number: <issue_number>) {
      subIssues(first: 50) { nodes { number title state url } }
      parent { number title state url }
    }
  }
}'
```

## Analyze context
- Extract the problem statement and acceptance criteria
- Summarize discussion history and blockers
- Identify affected files/modules
- Review sub-issues (if any) for work breakdown and progress

## Assess relevance
- Search codebase for mentioned APIs or modules
- Check related PRs and closed issues:
  ```bash
  gh pr list --search "in:title <keywords>" --json number,title,state,url
  gh issue list --state closed --search "<keywords>" --json number,title,url --limit 5
  ```

## Recommendation categories
- **Resolved**: recommend closing with rationale
- **Blocked**: note dependency and workarounds
- **Needs info**: ask targeted questions
- **Ready**: outline implementation path
- **Needs decision**: present options with trade-offs

## Draft comment
Create `.scratch/issue_comments/issue-<number>-evaluation.md` with:
```markdown
*This comment was generated with LLM assistance and may have been edited by the commenter.*

---

## Issue Evaluation
### Summary
<Restate issue + status>

### Current Relevance
- Code references checked:
- Related PRs:
- Changes since issue creation:

### Discussion Summary
<Key points from comments>

### Recommendation
- **Status:** <Ready / Needs info / Blocked / Can be closed>
- **Proposed next steps:** <Actionable items>
```

Present the draft for user approval before posting. Do not hard-wrap lines in markdown drafts; keep paragraphs on a single line.
