---
name: pr-to-green
description: Bring a pull request to green by syncing with main, resolving conflicts safely, and fixing failing checks with CausalPy conventions.
---

# PR to Green

This skill provides a deterministic maintainer workflow for taking an in-flight PR branch to green.

## When to use
- A PR is behind `main` and needs sync/rebase
- There are merge conflicts to resolve intelligently
- Remote checks are failing (tests, docs, lint, or packaging)
- You need a concise update for the PR describing what was fixed and what remains

## Preconditions
- Confirm branch and remotes before making git changes
- Preserve user/unrelated local modifications; never discard unknown work
- Use CausalPy environment commands:
  - `$CONDA_EXE run -n CausalPy <command>`
- For PyMC/PyTensor/matplotlib imports and test commands, request full permissions as needed

## PR number entrypoint (manual invocation)

If the user provides a PR number (for example `PR #724`), do this first:

1. Resolve PR metadata from GitHub
   - `gh pr view <number>` to get title, state, head branch, base branch, mergeability, and check summary.
   - `gh pr checks <number>` to get per-check pass/fail/pending status.
2. Move local repo to the PR branch
   - `gh pr checkout <number>` (preferred) so local branch tracks the PR head.
3. Verify branch context before edits
   - Confirm current branch, tracking status, and whether local uncommitted work exists.
4. Continue with the main workflow below using the PR base branch from GitHub metadata (not assumptions about `main`).

## Workflow

1. Triage and early exit
   - Assess PR state from entrypoint metadata: branch divergence, mergeability, and check results.
   - If the PR is up-to-date with its base branch, has no conflicts, and all remote checks pass:
     - Report "PR is green — no action needed" and stop.
   - Otherwise, classify what needs fixing: behind-base, conflicts, failing checks, or a combination.

2. Sync branch with base
   - Prefer `git fetch upstream` then `git rebase upstream/<base-branch>` (or maintainer-preferred merge strategy).
   - If the PR cannot be checked out, report the blocker and ask for maintainer guidance.
   - Resolve conflicts file-by-file with intent preservation.
   - Re-run targeted checks after conflict resolution to detect semantic drift.

3. Fix failing checks with smallest valid change
   - Lint/type: apply minimal code changes, avoid broad refactors.
   - Tests: patch root cause and add/adjust tests under `causalpy/tests/` if behavior changes.
   - Docs/doctest: follow docs placement and glossary/citation conventions.
   - Packaging/release checks: verify version and install/import expectations.

4. Re-run checks in escalating scope
   - Fast local signal first (targeted tests/lint).
   - Then full gate commands required by project norms.
   - Always run `pre-commit run --all-files` before final handoff.
   - If fixes introduced new failures, loop back to step 3 with the new failure set. Do not push until a full local pass is achieved or blockers are identified.

5. Push and verify remote
   - Push fixes to the PR branch (`git push`).
   - Monitor remote checks via `gh pr checks <number>` until they complete or a timeout is reached.
   - If remote checks fail on issues not reproducible locally, note the discrepancy explicitly.

6. Prepare maintainer-ready status update
   - What was failing.
   - What changed to fix it.
   - Which checks now pass (local and remote).
   - Remaining blockers (if any) and exact next steps.

## Subagent delegation

Use subagents (via the Task tool) when work is noisy, broad, or high-risk. Each subagent runs in its own context window and returns a condensed result.

### `ci-failure-investigator`

Trigger when CI logs are long/noisy, failures span multiple jobs, or failure family is unclear.

Handoff (include in Task prompt):
- PR number and branch name
- Failed job names and relevant log snippets or `gh run view` output
- Ask for: root cause ranking, fix-first order, and minimal patch plan

### `merge-conflict-analyst`

Trigger when rebase/merge produces non-trivial conflicts, intent differs across branches, or conflict count is high.

Handoff (include in Task prompt):
- Target branch and merge/rebase direction
- Full conflict list (`git diff --name-only --diff-filter=U`) and key conflict markers/snippets
- Ask for: conflict risk map, lowest-risk resolution order, and explicit escalations

### Maintainer escalation clause (required)

Do not auto-resolve and continue silently when conflicts are highly complex. Instead, stop and request maintainer input with a brief report.

Escalate by default when:
- `.ipynb` conflicts are messy (overlapping cell content plus metadata/output churn)
- conflict resolution would drop one branch's meaningful behavior
- conflicts touch core contracts and intent is ambiguous (experiments/models/tests/docs all changed around same behavior)

Escalation report must include:
- conflicted file list and risk class
- resolution options (1-2) with trade-offs
- recommended next step and what decision is needed from maintainer

## CausalPy guardrails
- Never use destructive git commands unless explicitly requested
- Do not create ad hoc test scripts; use `pytest` tests in `causalpy/tests/`
- Keep PyMC-heavy tests runtime-controlled via `sample_kwargs`
- Keep docs/notebooks in correct locations and formats
- If checks cannot be run, report exactly what was skipped and why
