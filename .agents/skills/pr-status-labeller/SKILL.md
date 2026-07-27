---
name: pr-status-labeller
description: Reconcile machine-computed status labels on open CausalPy PRs so the backlog state is visible on GitHub itself. Adds and removes only the `status:*` label namespace to match deterministic reality. Use when asked to refresh PR status labels, sync the triage labels, or run the labeller.
---

# PR Status Labeller

Keep a `status:*` label set on every open PR that mirrors its machine-computed
state, so anyone browsing GitHub sees the same triage picture the Discord
digest shows.

## Boundary

- Writes **only** the `status:*` namespace. Never touches `review:*`, `major`,
  `needs:maintainer-decision`, or any other human-owned semantic label.
- Never approves, merges, comments, or requests changes. Labels only.
- Never touches `-client` or any repo outside the configured one.
- Classification is the shared deterministic core in
  `../_shared/pr_triage/labels.py`. This skill does not decide states; it only
  reconciles labels against what the core computes.

## Managed labels

`status:conflicting`, `status:ci-failing`, `status:waiting-on-author`,
`status:ready-for-review`, `status:aging` (idle ≥30d), `status:stale`
(idle ≥90d). Drafts stay out of the reviewer-facing statuses; they only carry
`status:aging` / `status:stale` so WIP is never nagged.

## Workflow

1. Dry-run first, always:
   `python scripts/reconcile_labels.py --repo pymc-labs/CausalPy`
   This prints the exact add/remove diff per PR and writes nothing.
2. Review the diff. If it looks right, apply:
   `python scripts/reconcile_labels.py --repo pymc-labs/CausalPy --apply`
   `--apply` creates any missing managed labels, then reconciles each PR.

## Relationship to the digest

This skill and `pr-digest` share one core so they can never disagree. The
digest recomputes state live rather than reading these labels, so a missed
labeller run degrades GitHub visibility but never corrupts the digest. Run the
labeller on whatever cadence keeps GitHub fresh (e.g. daily); it is independent
of the Mon/Wed/Fri digest cadence.

## Cadence and safety

Because it mutates GitHub, treat this as a Tier 2 capability: start by running
the dry-run and eyeballing output for a few days before enabling `--apply` on a
schedule. The write surface is deliberately tiny (six labels, one namespace) so
a bad run is trivially reversible.
