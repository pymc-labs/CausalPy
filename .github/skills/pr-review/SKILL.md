---
name: pr-review
description: Review CausalPy pull requests — fetch context, run a structured pass over code, tests, docs, and process, then produce maintainer-quality comments for human review before posting. Use when the user asks to review a PR, check a PR, evaluate review feedback, or audit PR comments. Complements pr-to-green (which fixes PRs) and pr-workflows (which creates them).
---

# PR Review

This skill is for **reviewing** a CausalPy pull request — diagnosing what's there, judging it against project conventions, and producing actionable feedback. It is the read/diagnose/write-comment counterpart to `pr-to-green` (fix in-flight PR) and `pr-workflows` (create PR from issue).

## When to use

- The user asks to review, check, or evaluate a PR (e.g. "review #826", "look at PR 826", "what's outstanding on PR 826").
- The user asks whether a PR is ready to approve, merge, or land.
- The user asks to evaluate a contributor's response to prior review feedback ("did they address my comments?").
- The user asks to audit existing comments on a PR for relevance, accuracy, or remaining gaps.

Do **not** use this skill for:

- Fixing a failing PR — use [`pr-to-green`](../pr-to-green/SKILL.md).
- Turning an issue into a PR — use [`pr-workflows`](../pr-workflows/SKILL.md).
- Creating new issues that come out of a review — delegate to [`github-issues`](../github-issues/SKILL.md).

## Division of labour with automation

A pre-commit hook never misses; an agent review is judgement-heavy. Don't duplicate the hook's job. Mechanical checks already enforced by `prek` / CI (lint, format, mypy, schema validation, etc.) are not part of this skill — assume they pass and focus the human-equivalent attention on what they can't catch.

If during review you notice a *class* of issue that automation could catch but doesn't, file it as a follow-up issue via [`github-issues`](../github-issues/SKILL.md) rather than only flagging the one instance.

## Workflow

The full workflow is in [reference/workflow.md](reference/workflow.md). At a glance:

1. **Establish context** — fetch PR metadata, all commits, all reviews, all comments (review + issue threads), file changes.
2. **Read the relevant subset** — files changed, plus surrounding context for each change (base class for new subclass, sibling tests for new tests, etc.).
3. **Pass over the diff with the patterns checklist** — see [reference/what-to-look-for.md](reference/what-to-look-for.md). Group findings by severity.
4. **Verify claims** — if the contributor says "addressed all feedback", confirm against the actual code; if they say a rebase is clean, confirm against `git log`.
5. **Draft comments for human review** — never post directly. See [reference/posting-comments.md](reference/posting-comments.md).
6. **Post on approval** and report URLs back.

For batched PR work (multi-file diffs, parallel exploration), follow `pr-to-green`'s subagent delegation patterns — same `Task` triggers (`generalPurpose` for exploration, `ci-failure-investigator` for noisy CI logs) apply.

## What to look for

Severity-sorted patterns (must-fix → should-fix → nits) live in [reference/what-to-look-for.md](reference/what-to-look-for.md). The two domain-specific deep-dives are:

- [reference/code-patterns.md](reference/code-patterns.md) — `BaseExperiment` contract, `PyMCModel` vs. `RegressorMixin` dispatch, `_clone()` patterns, `Literal` for constrained strings, custom exceptions, `__repr__` style, memory-heavy retainers.
- [reference/docs-patterns.md](reference/docs-patterns.md) — notebook narrative quality, glossary linking, `:::{note}` admonitions, hide-input/hide-output cell tags, citations, sampler-output noise, helper-promotion judgement.

## Drafting comments

The full template is in [reference/posting-comments.md](reference/posting-comments.md). Two non-negotiable rules:

1. **Always draft for human review before posting.** Read the draft back and offer 2–3 framing variants (stricter / softer / approval-pending).
2. **Cite specific code locations.** Use line:line:filepath references when calling out code, not vague paraphrases.

## Mandatory pre-flight reads

Before reviewing any PR, the reviewing agent should have read or be able to load:

- `AGENTS.md` (root) — code structure, style, type hints, testing preferences.
- `CONTRIBUTING.md` — process expectations.
- `docs/source/notebooks/index.md` — toctree layout, where new notebooks should go.
- The PR's own description and any prior review iterations.

## How to extend this skill

When a review surfaces a new recurring pattern, add it. The process is in [reference/how-to-extend.md](reference/how-to-extend.md). Skill drift is real — every six months or so, prune patterns that have been formalised into hooks.

## Important behaviour

- Never post review comments without explicit human approval.
- Never approve, request changes, or merge a PR through `gh pr review --approve` unless explicitly instructed.
- When evaluating a contributor's response to prior feedback, verify against the code, not their summary. A "fixed all feedback" claim is a hypothesis to test, not a fact to accept.
- If a reviewer-bot identity has been used in the past on the PR (e.g. `claude-opus-4-7-xhigh`), preserve that attribution in posted comments so the human reviewer's voice and the agent's voice stay distinguishable.
- Do not duplicate work already covered by hooks/CI. If something is mechanically enforceable and not enforced, file an issue rather than re-flagging the instance.
