# PR Review Workflow

Use this workflow for maintainer-grade PR reviews. Steps 1-4 are read-only context gathering and analysis. Step 5 drafts feedback. Step 6 only happens after explicit human approval.

## Step 1: Establish Context

Gather the minimum context needed for the user's request. For full reviews, collect:

```bash
gh pr view <num> --json title,url,state,author,number,mergeable,mergeStateStatus,headRefName,baseRefName,commits,files,reviewDecision
gh api repos/:owner/:repo/pulls/<num>/commits --paginate --jq '.[] | {sha: .sha[0:8], msg: .commit.message | split("\n")[0]}'
gh api repos/:owner/:repo/pulls/<num>/reviews --jq '.[] | {user: .user.login, state, submitted_at, body}'
gh api repos/:owner/:repo/pulls/<num>/comments --paginate
gh api repos/:owner/:repo/issues/<num>/comments --paginate
gh pr diff <num> --name-only
gh pr checks <num>
```

If the user asks a narrow question, narrow the fetch. Do not pull every comment and full diff when the request is only to evaluate one latest response.

### Form a Review Brief

Before the diff pass, write a short internal brief:

- **Value hypothesis:** the two to four concrete outcomes the PR should deliver for users or maintainers.
- **Risk thesis:** the changed contract, invariant, or boundary with the highest cost if it is wrong.
- **Attack plan:** at least three plausible PR-specific failure modes, each with the evidence needed to falsify it.

Use the issue and existing PR comments as inputs to this brief. Add independent hypotheses; do not only repeat concerns already raised. For example, a canonical-container change should prompt probes for every backend producer, every direct consumer, multi-output or multi-unit behavior, and backwards-compatible callers.

## Step 2: Read the Relevant Subset

Read every changed file needed to evaluate the PR. Also read surrounding context:

- New subclass: read the base class and a sibling implementation.
- New test: read sibling tests and the production code it claims to cover.
- New experiment: read `BaseExperiment` and an existing experiment with the same backend shape.
- New PyMC model: read `PyMCModel` and a similar model's `_clone()` and prediction path.
- New check: read `causalpy/checks/base.py` and an existing check.
- New notebook: read a sibling notebook in the same docs section and `docs/source/notebooks/index.md`.
- New dataset loader: read nearby loaders and packaging expectations.

This is what catches omissions that are invisible in the diff alone, such as subclass overrides that drop base-class kwargs.

For each behavior-changing hunk, answer all of the following before moving on:

1. What previous behavior or invariant is this replacing?
2. Who produces the value this code assumes, and who consumes its result?
3. What is the weakest supported input or backend shape for this path?
4. Which test would fail if this hunk were removed or made subtly wrong?

When the answer to a consumer or producer is “all backends,” enumerate them. When the answer to the test question is “none,” assess whether missing coverage is a must-fix.

## Step 3: Walk the Checklist

Review against the universal checklist in `SKILL.md`, the PR-type resource files, and `review-patterns.md`. Group findings by severity:

- Must-fix: bugs, contract violations, dropped behavior, security issues, broken docs, missing tests for changed behavior.
- Should-fix: design concerns, unclear API shape, hidden runtime or memory costs, weak but not absent evidence.
- Nits: small clarity, naming, docstring, or cleanup suggestions that should not distract from substantive items.
- What worked well: specific positives that help the contributor preserve the strongest parts of the PR.

Do not stop at identifying a risk. Trace it to an observable failure, a violated contract, or evidence that the implementation already handles it. Where a concern is not a blocker, name the concrete follow-up or test that would improve confidence.

## Step 4: Verify Claims

Treat PR descriptions and contributor responses as hypotheses to test.

| Claim | Verification |
|---|---|
| "Rebased onto main" | Compare base and head history; check mergeability and stale/conflicting files. |
| "Addressed all feedback" | Walk each prior review item against the current branch. |
| "Tests added for the new behavior" | Read the tests and confirm they would fail without the change. |
| "Fixed the conflict" | Check GitHub mergeability and inspect the resolved conflict area. |
| "PR is small" | Inspect changed files, line counts, and logical themes. |
| "Docs build" | Confirm the relevant docs/notebook command was run or remote CI proves it. |

Flag incorrect claims respectfully and specifically. Vague disagreement is not useful review feedback.

For each claimed win, retain one piece of direct evidence from the diff, tests, benchmark, documentation, or runtime behavior. Use those verified wins in the executive summary; this prevents the final review from reducing the PR to a changelog or a CI report.

## Step 5: Draft Feedback

Draft comments for the user before posting anything to GitHub. For each finding, include:

- The affected file or code location.
- The behavior or risk.
- Why it matters.
- What should change.
- Whether it blocks approval or can be follow-up work.

Offer tone variants when helpful: as-written, stricter, or softer/follow-up.

## Step 6: Post Only on Approval

Only post after the user explicitly approves the exact comment or review action. Use HEREDOCs for multiline GitHub bodies:

```bash
gh pr comment <num> --body "$(cat <<'EOF'
<text>
EOF
)"
```

For formal reviews, never use `--approve` or `--request-changes` without explicit instruction. After posting, return the GitHub URL.
