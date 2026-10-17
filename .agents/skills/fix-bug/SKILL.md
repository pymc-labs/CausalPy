---
name: fix-bug
description: >-
  Autonomous bug-fix workflow. Guides the orchestrator through state detection,
  implementation, test/lint validation, and a bounded fix/review loop with an
  independent reviewer subagent. Use when fixing a GitHub bug report, or when the
  user says "fix bug", "bugfix", or references a bug issue or PR such as
  "fix bug #149", "bugfix #149", or "fix bug PR #123".
---

# Fix Bug — Autonomous Bug-Fix Workflow

Loop-engineered workflow for GitHub bugs. A **fixer** agent implements and pushes; a separate **reviewer** subagent reads the diff and posts PR comments. The fixer never self-reviews.

Read **Repo conventions** below (and `AGENTS.md` / `ARCHITECTURE.md` at the repo root) before changing code.

## Entry points

Invoke with either a **bug issue** or an **existing PR**. The agent detects where work left off and continues from there.

| You say | What the agent does |
|---------|---------------------|
| `fix bug #149` / `bugfix #149` | Load issue #149 → find or create a PR → implement or resume |
| `fix bug PR #123` / `bugfix pr 123` | Load PR #123 → derive linked issue if present → resume from PR state |

**Fresh issue** (no PR yet): read the bug report, reproduce or trace root cause, implement on a fix branch, open a PR, then enter the review loop.

**Existing PR** (work already started): check out the PR branch, read issue + PR context, then pick up at the right phase — fix failing CI, address an unaddressed reviewer round, or spawn the next review pass if CI is green.

Both paths converge on the same PR-centric review loop (Phase 3).

## Repo conventions

Customize this section when copying the skill into a new repository. The orchestration phases below assume these conventions exist somewhere (here or in `AGENTS.md`).

- **Agent guide**: `AGENTS.md` or `CONTRIBUTING.md` at repo root — architecture, style, and workflow rules.
- **Environment**: how to run commands (e.g. `uv run`, `npm run`, `poetry run`). Never use system Python/Node unless the repo does.
- **Tests**: targeted test command for the touched area; full or fast suite command before PR is ready.
- **Lint**: formatter/linter/typecheck command(s) required before push.
- **Branch naming**: e.g. `fix/<issue-number>-<short-slug>`.
- **Commit messages**: imperative mood; `Fixes #N` in body when closing a bug issue.
- **PR body**: `## Issue links` with accurate closing semantics (`Fixes` / `Part of` / `Related to`), summary, test plan.
- **Fixer PR comments**: brief comments explaining what changed and why (see Phase 2).
- **Test policy**: whether existing test assertions may be changed (many repos: add tests, don't change expected behavior without human review).
- **Escalation label**: label applied when the loop exhausts (e.g. `needs developer attention`).

### CausalPy (this repo)

- **Agent guide**: `AGENTS.md` at repo root; read `ARCHITECTURE.md` before core code changes. Do **not** load `CONTRIBUTING.md` by default.
- **Environment**: `$CONDA_EXE run -n CausalPy <command>` for commands that import project code, run tests, build docs, or use repo tooling. Never use `$CONDA_EXE activate`. Request full permissions (`all`) for `pytest`, `make doctest`, or any command importing PyMC, PyTensor, or matplotlib (sandbox blocks their cache paths).
- **Tests**: `$CONDA_EXE run -n CausalPy python -m pytest causalpy/tests/test_<module>.py -x -v` for targeted; full suite via `$CONDA_EXE run -n CausalPy python -m pytest`. PyMC-heavy tests must use minimal `sample_kwargs` to keep runs fast.
- **Lint**: `prek run` on changed files during iteration; `prek run --all-files` before handoff. Run `make test-patch-cov` at PR-ready milestones when production code changed.
- **Branch naming**: `fix/<issue-number>-<short-slug>` (e.g. `fix/149-plot-label-regression`).
- **Do not modify existing test assertions** unless adding new tests or removing obsolete ones.
- **Escalation label**: `needs:maintainer-decision`.

## Phase 1: State detection

Resolve the work item, then determine current state.

### Resolve issue and PR

```bash
# --- Entry: issue number ---
gh issue view $ISSUE_NUMBER --json title,body,state,labels,comments

# Find an existing PR for this issue (search title/body, not only "Fixes #")
gh pr list --search "#$ISSUE_NUMBER" --state all --json number,headRefName,state,title

# --- Entry: PR number (skip if you already have ISSUE_NUMBER) ---
gh pr view $PR_NUMBER --json number,title,body,state,headRefName,comments,statusCheckRollup,labels

# Linked issue: parse "Fixes #N" / "Closes #N" from PR body, or ask the user if absent
```

If started from a PR, check out its head branch before changing code:

```bash
gh pr checkout $PR_NUMBER
```

### Inspect PR state

Round markers use the prefix `fix-bug-round` (HTML comments). Fixer summaries use `fix-bug-fix-summary`. Approval uses `fix-bug-approved`. Use the same prefixes in every repo copy of this skill so resume logic stays portable.

```bash
# Count review-round markers from automated reviewer:
gh pr view $PR_NUMBER --json comments -q \
  '[.comments[].body | select(test("fix-bug-round"))] | length'

# Approval marker (distinct from reviewer crash — no comment at all):
gh pr view $PR_NUMBER --json comments -q \
  '[.comments[].body | select(test("fix-bug-approved"))] | length'

# CI status:
gh pr checks $PR_NUMBER

# On resume: recover chosen scope from the latest fixer summary
gh pr view $PR_NUMBER --json comments -q \
  '[.comments[].body | select(test("fix-bug-fix-summary"))] | last'
```

| Detected state | Action |
|----------------|--------|
| Issue closed or PR merged | Exit — already resolved |
| Human pushed or commented since last automation activity | Escalate — don't overwrite human work |
| PR has `fix-bug-approved` and CI passing | Exit — review loop complete (run umbrella wrap-up if applicable) |
| No PR exists (issue entry only) | Full flow: understand → implement → push → create PR → review loop |
| PR exists, CI failing | Read failures → fix → push → wait CI → continue |
| PR exists, unaddressed review comment (has round marker) | Address findings → push → wait CI → continue |
| PR exists, CI passing, no unaddressed review | Enter review loop (spawn reviewer) |
| PR exists, resuming | Read latest `fix-bug-fix-summary` comment to recover scope before changing code |
| 3+ round markers already present | Escalate immediately |

Match rows **top-to-bottom**. Human intervention must win over `fix-bug-approved` — e.g. a human commenting after approval should escalate, not exit silently.

**Fallback when round markers are missing**: estimate rounds from PR comments that are **not** orchestration markers (`fix-bug-fix-summary`, `fix-bug-escalation`, `fix-bug-approved`). Do not count fixer summaries — they are posted every push and would trigger premature escalation.

```bash
marker_count=$(gh pr view $PR_NUMBER --json comments -q \
  '[.comments[].body | select(test("fix-bug-round"))] | length')
fallback_count=$(gh pr view $PR_NUMBER --json comments -q \
  '[.comments[].body | select(test("fix-bug-fix-summary|fix-bug-escalation|fix-bug-approved") | not)] | length')
# round_estimate = max(marker_count, fallback_count); if ambiguous, treat as round 0
```

### Scope selection (umbrella / meta issues)

Some issues track many bugs (sub-issues, checklists, or long itemised lists). **Fix one thing per PR** — never attempt the whole umbrella in a single pass.

**Detect umbrella issues** before implementing:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

# Sub-issues (primary — GitHub sub-issues API)
gh api "repos/$REPO/issues/$ISSUE_NUMBER/sub_issues" \
  --jq '.[] | {number, title, state}'

# Issue body and metadata (checklists, cross-references)
gh issue view $ISSUE_NUMBER --json title,body,state,labels,comments
# Secondary: scan body for unchecked `- [ ]` items, numbered bug lists, or `#NNN` cross-refs
```

Placeholders in templates below use angle brackets (`#<scope-issue>`, `#<parent-issue>`) — substitute the actual issue numbers. Heredocs use `<<'EOF'` (no shell expansion).

| Issue shape | Pick exactly one |
|-------------|------------------|
| Open sub-issues | Highest-priority open child (priority labels, then smallest/isolated fix) |
| Closed sub-issues + open parent | Next open child, or one unchecked checklist item on the parent |
| Checklist / itemised list, no sub-issues | One unchecked item — prefer the most obvious or self-contained |
| Single clear bug | The whole issue (normal case) |

**Tie-breaking**: When several candidates look equally valid, pick one and proceed — do not ask the user. The user delegated this work; a coin-flip question blocks progress. Use the first open sub-issue by number, or the first unchecked checklist item top-to-bottom, and record the choice in the fixer summary.

**Record scope explicitly** before coding:

- Note the scope issue number (`#<scope-issue>`) — the child issue when fixing a sub-issue, otherwise the issue you were pointed at.
- Note which checklist item or sub-issue you chose and why (one sentence).

**Closing semantics** (must match what the PR actually does):

| Situation | PR body / commit footer |
|-----------|-------------------------|
| Fully fixes one issue | `Fixes #N` |
| Fixes one child of an umbrella parent | `Fixes #child` and `Part of #parent` |
| Partial progress on a single issue (one checklist item) | `Part of #N` — do **not** use `Fixes #N` on the parent |
| Investigation only, no fix yet | `Related to #N` — rare; prefer not opening a PR |

When in doubt, under-close (`Part of`) rather than over-close (`Fixes` on a parent meta issue).

## Phase 2: Fix implementation

Skip steps already done when resuming an existing PR (e.g. branch exists, partial fix landed).

1. **Read the bug report thoroughly** — issue body, all comments (especially triage bot comments), linked PR discussion, any linked docs. If the issue is an umbrella, complete **Scope selection** above and fix only the chosen item.
2. **Identify root cause** — trace the actual code path; reproduce with a failing test when possible. Don't guess from the description alone.
3. **Consider 2–3 approaches** — pick the smallest correct diff. Prefer fixing the shared function once over patching each caller.
4. **Implement** — follow the repo's agent guide and style conventions.
5. **Validate** — run targeted tests and lint per **Repo conventions**.
6. **Commit and push**:

   ```bash
   git add -A
   git commit -m "$(cat <<'EOF'
   Short imperative summary

   Fixes #<scope-issue>. Part of #<parent-issue> if umbrella. Why this approach.
   EOF
   )"
   git push -u origin HEAD
   ```

7. **Create PR** (only if none exists yet):

   ```bash
   gh pr create --title "Fix: <short description>" --body "$(cat <<'EOF'
   ## Summary
   <1-2 sentences on the approach.>

   ## Issue links
   - Fixes #<scope-issue> (or Part of #<parent-issue> — match actual scope)
   - <If umbrella: which sub-issue or checklist item this addresses>

   ## Test plan
   - [ ] Targeted tests pass
   - [ ] Full/fast test suite passes
   - [ ] Lint passes
   EOF
   )"
   ```

8. **Post a fixer summary comment** on the PR (required after initial push and after each round of review fixes):

   ```bash
   gh pr comment $PR_NUMBER --body "$(cat <<'EOF'
   <!-- fix-bug-fix-summary -->
   ## Fix summary

   **Scope**: Fixes #<scope-issue> / Part of #<parent-issue> — <which sub-issue or checklist item, if umbrella>
   **Root cause**: <one sentence>
   **Approach**: <why this fix, not an alternative>
   **Not in scope**: <what was deliberately left for a follow-up PR>
   EOF
   )"
   ```

   Keep it brief (4–6 sentences). The reviewer reads the diff; this comment explains intent for humans and the next automation pass.

9. **Update PR body when scope drifts** — if implementation narrows or shifts scope, edit **only** the `## Issue links` section. Read the current body first; never replace the whole description (that would destroy human edits to Summary, Test plan, or checkboxes):

   ```bash
   gh pr view $PR_NUMBER --json body -q .body   # read current body
   # Merge: keep all existing sections; update only ## Issue links
   gh pr edit $PR_NUMBER --body "$(cat <<'EOF'
   <paste merged body — preserve everything outside Issue links>
   EOF
   )"
   ```

## Phase 3: Review loop

### Spawning the reviewer subagent

Spawn a Task subagent with these characteristics:

- **Fresh context**: The reviewer has NO knowledge of your fix reasoning.
- **Prompt**: Include the PR number, the diff (or instruct it to read via `gh pr diff`), and the review criteria below.
- **Role boundary**: The reviewer NEVER modifies code. Its only output is a PR comment.

### Reviewer prompt template

> You are an independent code reviewer for this repository. You have never seen this code before.
>
> **Your job**: Review PR #$PR_NUMBER. Post ONE review comment on the PR with your findings. You NEVER modify code or push. Permitted reads only:
>
> - `gh pr diff $PR_NUMBER`
> - `gh pr view $PR_NUMBER --json body,comments`
> - The repo's `AGENTS.md` / `CONTRIBUTING.md` if present
>
> **Read fixer intent** (if present): scan PR comments for `<!-- fix-bug-fix-summary -->`. Use for context only — if the diff and the summary disagree, the diff is the truth and the disagreement is a 🔴 finding.
>
> **Review criteria** (check each):
>
> - Correctness: Does the fix address the root cause? Any logic errors?
> - Scope alignment: Does the diff match the PR **Issue links** section and the fixer summary? Flag over-closing (`Fixes #parent` when only one sub-item is done) or under-documented scope.
> - Regressions: Could this break existing behavior? Check callers of modified functions.
> - Edge cases: Are boundary conditions handled?
> - Test coverage: Are the new/changed paths tested?
> - Style: Matches repo conventions (formatting, types, naming, comments).
> - Performance: No unnecessary hot-path regressions.
> - Error messages: Clear and actionable where the repo cares about them.
>
> **Severity levels**:
>
> - 🔴 **Must fix**: Correctness bug, regression, or missing guard. Blocks merge.
> - 🟡 **Should fix**: Missing test, unclear naming, style violation.
> - 🟢 **Nitpick**: Optional improvement, not blocking.
>
> **Output format**: Post a PR comment via `gh pr comment $PR_NUMBER --body "..."`.
> Start the comment with the marker: `<!-- fix-bug-round:$ROUND -->`
> If you find NO actionable issues (no 🔴 or 🟡), post an approval comment with marker:
> `<!-- fix-bug-approved -->`
> Reviewed — no actionable issues found. Approving.
>
> Absence of any comment means reviewer failure, not approval.
>
> Be specific and actionable. Another agent will read your comments and fix what you raise.

### Loop control

```text
round = count_existing_markers()  # or internal counter during live session

while round < 3:
    wait_for_ci()  # gh pr checks --watch $PR_NUMBER
    spawn_reviewer(round + 1)
    verdict = read_reviewer_output()

    if verdict == "approved" (fix-bug-approved marker posted):
        break  # Done — run umbrella wrap-up if applicable

    address_review_findings()
    push_fixes()
    round += 1

if round >= 3 and not approved:
    escalate()
```

### Addressing review findings

- Read the reviewer's comment carefully.
- Address all 🔴 (must fix) and 🟡 (should fix) items.
- 🟢 (nitpick) items: fix if trivial, skip if not.
- Run tests and lint before pushing.
- Post a **fixer summary comment** (step 8 above) describing what you changed in response to the review — do not push silently.
- Do NOT post your own code review or approve your changes — the independent reviewer will check on the next pass.

### Umbrella wrap-up (on approval)

When the review loop completes with `fix-bug-approved` and the work scoped to one item of an umbrella issue:

1. **Tell the user** which item was fixed and list what remains (open sub-issues or unchecked checklist items on the parent).
2. **Optionally comment on the parent issue** so the tracker shows partial progress:

   ```bash
   gh issue comment $PARENT_ISSUE --body "$(cat <<'EOF'
   Automated fix PR #<pr-number> merged/ready — addressed: <item fixed>.
   Remaining: <enumerate open sub-issues or unchecked items>.
   EOF
   )"
   ```

   One item per PR is intentional; the user invoked `fix bug` on the umbrella and should see clear next steps, not a silent partial completion.

## Phase 4: Escalation

When the loop is exhausted (3 rounds) or a safety cap is hit:

```bash
# Label the PR (use escalation label from Repo conventions)
gh pr edit $PR_NUMBER --add-label "$ESCALATION_LABEL"

# Label the linked issue (when known)
gh issue edit $ISSUE_NUMBER --add-label "$ESCALATION_LABEL"

# Post summary comment
gh pr comment $PR_NUMBER --body "$(cat <<'EOF'
<!-- fix-bug-escalation -->
## Automated fix escalation

This PR has been through 3 review/fix rounds without full resolution.
Remaining findings from the last review are above.

**Handing off to a developer.** The automated agents were unable to fully
resolve the reviewer's concerns. Please review the outstanding items
and either fix or close as acceptable.
EOF
)"
```

## Safety caps (hard limits)

These apply regardless of marker state:

| Cap | Limit | On breach |
|-----|-------|-----------|
| Reviewer spawns per session | 3 | Escalate |
| Total pushes per session | 6 | Escalate |
| Single CI wait | 30 minutes | Post timeout note, exit |
| Consecutive reviewer failures | 2 | Escalate |

## Robustness notes

- **Internal counter is primary** during a live session. Markers are for observability and resume.
- **On resume with missing round markers**: use the fallback query above (excludes fixer summaries). If ambiguous, start at round 0.
- **Reviewer subagent failure**: If it returns without posting a comment, check the PR. No comment = failed round. A `fix-bug-approved` comment = success.
- **Human intervention**: If a human has pushed commits or posted comments since the last automation activity, escalate rather than overwriting.
- **Already resolved**: If the issue is closed or PR is merged, exit immediately.
- **PR without linked issue**: Workflow still runs; omit issue labels/steps that need `ISSUE_NUMBER` or ask the user for the bug issue number.
- **Umbrella issues**: One bug per PR. Re-read `fix-bug-fix-summary` comments on resume to recover chosen scope.
- **Scope ties**: Pick one candidate and document the choice — never block on a user tie-break.
- **Over-closing**: Never put `Fixes #N` on a parent meta issue unless this PR resolves every tracked item. Prefer `Part of #N` plus `Fixes #child`.

## Copying to another repo

1. Copy `.agents/skills/fix-bug/SKILL.md` unchanged except **Repo conventions** — replace the CausalPy subsection with that repo's commands and policies (or a single pointer to its `AGENTS.md`).
2. Keep marker prefixes (`fix-bug-round`, `fix-bug-escalation`, `fix-bug-fix-summary`, `fix-bug-approved`) identical so cross-repo tooling and resume logic stay consistent.
3. Adjust escalation label name only in **Repo conventions** if the target org uses a different label.
