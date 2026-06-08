# Review Comments

Use this resource when drafting, presenting, or posting review comments.

## Non-Negotiable Rules

- Never post without explicit human approval.
- Never approve, request changes, or merge unless the user explicitly asks for that action.
- Cite specific files and code locations when calling out code.
- Use HEREDOCs for multiline GitHub comments so markdown renders correctly.
- Preserve the distinction between the maintainer's voice and agent-authored review text.

## Comment Shapes

Use PR issue-thread comments for:

- Multi-item review summaries.
- Replies to existing general comments.
- Process or scope feedback.
- Status updates such as "verified the rebase" or "confirmed feedback addressed."

Use inline review comments for:

- Specific code-level suggestions tied to a line.
- Bugs where the exact location is central to the finding.

Use formal reviews only with explicit approval:

- `gh pr review --comment` for a neutral submitted review.
- `gh pr review --approve` only when the user explicitly instructs approval.
- `gh pr review --request-changes` only when the user explicitly instructs a request-changes review.

## Multi-Item Review Summary

```markdown
Thanks for the PR. I have comments grouped by severity below.

## Must-fix

### 1. [Concise title] (`path/to/file.py`)

[Two to four sentences explaining the issue, why it matters, and what should change.]

## Should-fix

### 2. [Concise title]

[Design or evidence concern. Say whether this must happen in the current PR or can be follow-up work.]

## Nits

### 3. [Concise title]

[Small cleanup, clarity, naming, or docstring item.]

## What worked well

- [Specific positive about the implementation, tests, docs, or scope.]
- [Another specific positive when available.]
```

Keep positives specific. Generic praise adds little, but concrete reinforcement helps contributors preserve the good parts while fixing issues.

## Status Update on a Contributor Response

```markdown
Thanks @<contributor> - I checked the latest branch against the prior review items.

Confirmed addressed:

- [Specific item and evidence.]
- [Specific item and evidence.]

Still open:

- [Item, current state, and exact ask.]

Merge readiness: [green / blocked / waiting on CI / needs maintainer decision].
```

## Inline Suggestion Shape

```markdown
## Suggestion: [concise title]

[Briefly describe the current code and why the suggestion matters.]

### Suggested shape

1. [Option A.]
2. [Option B.]

I'd lean toward [option] because [reason].

Scope: [current PR / follow-up is fine / maintainer decision needed].
```

## Correction

If you posted something incorrect, correct it promptly instead of adding ambiguity.

```markdown
Correction to my previous comment: [wrong claim] was incorrect because [reason]. The accurate version is [correct statement]. [Say whether any original ask still stands.]
```

## Posting Commands

```bash
gh pr comment <num> --body "$(cat <<'EOF'
<text>
EOF
)"
```

After posting, return the URL from GitHub.
