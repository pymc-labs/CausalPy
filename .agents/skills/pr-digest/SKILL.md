---
name: pr-digest
description: Produce the Mon/Wed/Fri CausalPy PR triage digest for Discord — a read-only, recommendation-only summary that surfaces exactly one hard decision, a short clean-review queue, and everything else collapsed as "no action needed". Use when asked to generate the PR digest, summarize the PR backlog for a maintainer, or run the triage recap.
---

# PR Digest

Generate the maintainer-facing PR triage digest. This is **Tier 1**: read-only.
The digest recommends; it never approves, merges, comments, labels, or mutates
GitHub in any way. Every action it names is the maintainer's to take.

## Boundary

- This skill only reads and reports. It never writes to GitHub. Writing
  `status:*` labels is a separate concern owned by the `pr-status-labeller`
  skill; mechanical GitHub actions (nudges, branch syncs) are Tier 2 and not
  part of this skill.
- Classification is deterministic and lives in
  `../_shared/pr_triage/labels.py`. Do not re-implement it or let the model
  re-derive states by eye — always run the script so the digest matches the
  labeller byte-for-byte.
- The only judgment this skill adds is prose: the framing of the single hard
  item and any cross-PR clustering insight. Everything else is templated.

## Workflow

1. Run the builder:
   `python scripts/build_digest.py --day <Mon|Wed|Fri> --mention <discord_user_id>`
   It calls the shared core live (so it is never stale), selects the one hard
   item, groups the rest, and emits JSON with a ready-to-post `markdown` field
   plus a `hard_item` payload.
2. Fill the `{{HARD_ITEM_FRAMING}}` placeholder in `markdown` following
   `resources/format.md`. This is the one place light judgment is required.
3. Post the completed markdown to the CausalPy Discord channel, tagging the
   maintainer. Do nothing else — no GitHub writes.

## The one-hard-item rule

Confirmed ceiling: surface exactly **one** deep-focus decision per digest.
Maintainers realistically do no more than one per day, so a longer list just
gets deferred wholesale (choice overload). The builder picks it deterministically
(oldest decision-needed or major-ready PR); remaining hard items are listed as
"queued behind this one" so they are visible but not demanded today.

## Framing the hard item (the anti-paralysis layer)

The research this is built on: choice overload, the default effect, omission
bias / loss aversion, and implementation intentions. The digest counteracts
decision paralysis by three moves, all applied in `resources/format.md`:

- **Recommend a default, don't ask an open question.** The maintainer vetoes or
  adjusts a proposed direction rather than deciding from scratch.
- **Frame every hard item as a reversible, small next step**, never a terminal
  architectural commitment. "State a direction so contributors can move" beats
  "decide the API forever".
- **Name the cost of *not* deciding** (contributor idle days, downstream
  blockage), so loss aversion pushes toward action, not just away from a wrong
  call.

## Cadence

Monday, Wednesday, Friday. Triggered either by a Daimon automation that runs
this skill and posts, or by a maintainer running it locally. The logic is
identical across triggers; only the posting step differs.
