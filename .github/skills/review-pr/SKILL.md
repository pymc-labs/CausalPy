---
name: maintainer-pr-review
description: Review CausalPy pull requests end-to-end by classifying PR type, checking branch freshness, mergeability, remote CI, correctness, security, tests, docs, and maintainer concerns. Use when asked to review a PR, assess a branch before merge, summarize PR risks, or request changes.
---

# Maintainer PR Review

Use this skill to evaluate whether a PR is correct, safe, understandable, and merge-ready. This is a review workflow, not primarily a fix workflow.

## Boundary

- If the user asks to review, assess, summarize risks, or decide whether a PR is ready to merge, use this skill.
- If the user asks to make the PR green by fixing CI, conflicts, or review comments, use `pr-to-green` for CausalPy-specific greening work.
- For continuous merge-readiness monitoring, repeat this skill's intake, CI, and comment checks on a cadence. If the repository later adds a dedicated monitoring skill, prefer that.
- If the review uncovers clear, small fixes and the user asked you to fix them, keep changes scoped to the PR's intent and follow the repo's commit and `prek` rules.
- Never post review comments, approve, request changes, or merge through GitHub without explicit human approval.
- Do not duplicate mechanical checks already covered by hooks and CI. If a recurring issue is mechanically enforceable but not enforced, recommend a follow-up issue instead of treating each instance as bespoke review work.

## Intake

Follow `resources/workflow.md` for the full workflow. At a glance:

1. Identify the PR number or current branch, base branch, head branch, and whether the branch tracks a remote.
2. Inspect the local working tree before any git operation and preserve unrelated local changes.
3. Fetch PR metadata, commits, reviews, issue comments, changed files, mergeability, and check summary.
4. Check whether the branch is behind its base and whether GitHub reports conflicts. Do not resolve conflicts as part of review unless explicitly asked; report conflict risk and recommend `pr-to-green` when needed.
5. Check remote CI with `gh pr checks` or the equivalent GitHub command. Distinguish failed, pending, skipped, and missing required checks.
6. Inspect the full PR diff against the base branch, not only the latest commit.
7. Verify contributor claims against code, tests, and branch history before accepting them.

## Classify the PR

Classify the PR by its dominant risk profile, then read the matching resource file. For mixed PRs, read every relevant resource before reviewing.

- Feature implementation: read `resources/pr-type-features.md`.
- Bug fix: read `resources/pr-type-bug-fixes.md`.
- Refactor: read `resources/pr-type-refactors.md`.
- Docs or notebooks: read `resources/pr-type-docs-notebooks.md`.
- Data or dataset changes: read `resources/pr-type-data-datasets.md`.
- Tests, CI, packaging, or infrastructure: read `resources/pr-type-tests-ci-infra.md`.

When classification is unclear, state the likely categories and review against the stricter applicable checklist.

## Deep Dives

Read these when the PR touches the relevant surface:

- CausalPy source-code conventions: `resources/code-patterns.md`.
- Documentation and notebook conventions: `resources/docs-patterns.md`.
- Severity-sorted recurring review patterns: `resources/review-patterns.md`.
- Drafting or posting review comments: `resources/review-comments.md`.
- Updating this skill with recurring patterns: `resources/maintenance.md`.

## Universal Checks

- Correctness: the implementation matches the PR's stated intent, handles important edge cases, and does not introduce silent behavior changes.
- Security and privacy: no secrets, credentials, tokens, private data, unsafe deserialization, command injection, path traversal, or unnecessary network access.
- Causal/statistical accuracy: causal claims, model assumptions, estimands, priors, simulations, and examples are technically accurate and not overstated.
- Public API: released APIs remain compatible unless the PR intentionally changes them and documents the change; new public APIs have explicit signatures and documentation.
- Tests: behavior changes have meaningful tests in `causalpy/tests/`; PyMC-heavy tests use runtime-controlled `sample_kwargs`; no throwaway verification scripts are added.
- Docs: user-facing behavior changes have docs or examples where appropriate; docs follow CausalPy notebook, MyST, glossary, and citation conventions.
- Dependencies and packaging: new dependencies are justified, declared in the right place, and do not duplicate existing tools.
- Performance and runtime: expensive sampling, notebook execution, data loading, and CI changes are bounded and justified.
- Maintainability: the change follows local patterns, avoids broad unrelated refactors, and keeps ownership boundaries clear.

## CausalPy Review Norms

- Before reviewing code, read `AGENTS.md` and relevant local context. For docs-heavy PRs, also inspect `docs/source/notebooks/index.md`; for process-sensitive PRs, inspect `CONTRIBUTING.md` when present.
- Use `$CONDA_EXE run -n CausalPy <command>` for commands that import project code, run tests, build docs, or invoke repo tooling. `AGENTS.md` defines how to detect or set `CONDA_EXE`; if it is unset, inspect that environment guidance before running project commands.
- Use full permissions for commands that import PyMC, PyTensor, or matplotlib to avoid false sandbox failures.
- During review, prefer targeted local checks that match the changed surface. If you edit code or prepare a commit, run `prek run` during iteration and `prek run --all-files` before handoff unless the user explicitly says not to.
- For markdown-only skill or docs changes, a structural read-back may be enough; report when full checks were not run and why.

## Review Output

Lead with findings, ordered by severity. If there are no findings, say so clearly and mention residual risk or checks not run.

Use this structure:

```markdown
## Findings
- [severity] `path`: issue, why it matters, and what should change.

## Merge Readiness
Verdict: approve / request changes / blocked / needs maintainer decision.
Branch status: up to date or behind base; conflicts if any.
CI status: green, failing, pending, skipped, or unavailable.

## PR Summary
One short paragraph describing what the PR changes and why.

## Test Evidence
List local and remote checks observed. Include commands only when they were actually run.

## Open Questions
Only include questions that affect merge readiness or review confidence.
```

When drafting comments for posting, show the draft to the user first and wait for approval. Preserve the distinction between the human maintainer's voice and any agent-authored review text.
