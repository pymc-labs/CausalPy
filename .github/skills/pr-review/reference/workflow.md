# PR Review Workflow

Six steps. Steps 1–4 are read-only context-gathering and analysis. Step 5 is drafting. Step 6 only happens with explicit human approval.

## Step 1 — Establish full context

Run these in parallel where possible:

```bash
# PR metadata
gh pr view <num> --json title,url,state,author,number,mergeable,mergeStateStatus,headRefName,baseRefName,commits,files

# All commits
gh api repos/:owner/:repo/pulls/<num>/commits --paginate --jq '.[] | {sha: .sha[0:8], msg: .commit.message | split("\n")[0]}'

# All formal reviews (CHANGES_REQUESTED, COMMENTED, APPROVED)
gh api repos/:owner/:repo/pulls/<num>/reviews --jq '.[] | {user: .user.login, state, submitted_at, body}'

# All inline review comments (line-anchored)
gh api repos/:owner/:repo/pulls/<num>/comments --paginate

# All issue-thread comments (general PR conversation)
gh api repos/:owner/:repo/issues/<num>/comments --paginate

# Files changed
gh pr diff <num> --name-only
```

If the user has already provided partial context (e.g. "check the last comment"), narrow the fetch — don't re-pull everything.

## Step 2 — Read the relevant subset

Read every file in the diff. For each, also pull surrounding context:

- Subclass added → read the base class.
- New test → read sibling tests in the same file.
- New experiment class → read `BaseExperiment` and at least one existing experiment.
- New check → read `causalpy/checks/base.py` and an existing check.
- Notebook added → read at least one sibling notebook in the same docs section.

This is what catches "this `_clone()` doesn't forward `priors=self._user_priors` like the base class does" — you can't notice the omission without reading the base class.

## Step 3 — Pass over the diff

Walk the patterns checklist in [what-to-look-for.md](what-to-look-for.md) explicitly. Don't trust unaided memory; the checklist exists because reviews otherwise drift toward whatever's most recently in mind.

Group findings by severity:

- **Must-fix** — bugs, contract violations, dropped behaviour, missing required forwarding (e.g. base-class kwargs).
- **Should-fix** — design judgement (default values, API ergonomics, memory footprint), missing tests for behavioural changes, broad except clauses.
- **Nits** — style, docstring polish, naming, defensive code that isn't actually defending anything.
- **Things you liked** — actually include this. Reviews that only criticise are demoralising and lose signal.

## Step 4 — Verify claims

If the contributor (or a previous review iteration) claimed something is done, verify against current code. Common claim shapes and their tests:

| Claim | Verification |
|---|---|
| "Rebased onto main" | `git log --oneline base..head` shows merge commits / no problematic commits |
| "Addressed all feedback" | Walk each review item against current files |
| "Tests added for the new behaviour" | Read the test, confirm it would fail without the fix |
| "Fixed the conflict" | `gh pr view --json mergeable` shows `MERGEABLE` |
| "PR is small" | `gh pr view --json files` line-count |

Any claim that turns out to be wrong should be flagged respectfully but specifically — vague disagreement is unhelpful.

## Step 5 — Draft comments for human review

Never post directly. Always:

1. Lay out findings in chat for the user.
2. Show the draft comment text in a code block.
3. Offer 2–3 framing variants (stricter, softer, approval-pending) so the user picks the tone.
4. Wait for explicit approval.

See [posting-comments.md](posting-comments.md) for full templates.

## Step 6 — Post on approval

Only after the user explicitly approves:

```bash
gh pr comment <num> --body "$(cat <<'EOF'
<text>
EOF
)"
```

For inline review comments anchored to a line, use `gh api repos/:owner/:repo/pulls/<num>/comments` with the appropriate JSON body. For a formal review (with `event: COMMENT`/`APPROVE`/`REQUEST_CHANGES`), use `gh pr review` — but never `--approve` or `--request-changes` without an explicit user instruction.

After posting, return the URL.

## Loops and follow-up

If the review surfaces:

- **A formalisable pattern** (could be a hook) → file a follow-up issue via [`github-issues`](../../github-issues/SKILL.md).
- **A code-level fix the contributor should make** → leave a PR comment with the fix sketched.
- **A scope concern** (PR too big, mixes themes) → flag in the review summary and reference past examples (`#826` "mega payload" comment is the canonical example here).
- **A merge blocker** (conflicts with a recently-merged PR) → identify the colliding commits and propose the rebase.
