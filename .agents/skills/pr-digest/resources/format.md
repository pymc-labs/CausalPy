# Digest format and the hard-item framing

The builder emits nearly-complete markdown. Your only writing task is the
`{{HARD_ITEM_FRAMING}}` block for the single hard item. Keep everything else
as the builder produced it.

## Structure (builder-owned, do not restructure)

1. Header + one-line "today you have N things" summary + the standing note that
   the digest never acts on its own.
2. 🎯 **One decision today** — the single hard item (you frame this).
3. 🟢 Approved, needs a merge click (only if any).
4. ✅ Quick reviews — clean reads, external contributors first.
5. ━━━ divider ━━━ then the FYI zone: waiting-on-author, mechanical,
   stale/aging drafts, in-flight drafts. All counts + links, no action.

## Writing the hard item

Aim for three short paragraphs. Apply all three anti-paralysis moves.

1. **Context + why it's stuck.** Explain what the PR(s) actually decide and why
   it has stalled. If several PRs are one underlying decision (e.g. packaging:
   pixi + conda-lock + flit), cluster them explicitly — the builder only names
   one lead PR, so you add the "these three are really one decision" insight
   and link the others.
2. **Recommended direction (veto or adjust).** State a concrete default the
   maintainer can accept in one word or override. Not an open question.
3. **Reversible + cost of waiting.** Say plainly that the step is small and
   reversible (stating a direction, not merging), then quantify the cost of
   inaction (contributor idle days, downstream PRs blocked).

### Example (packaging cluster)

> **Packaging & env tooling: pick a coherent direction.** #239 (setuptools →
> flit), #412 (add pixi) and #281 (conda-lock) all poke the same packaging
> surface, and that coupling is why each has sat for years — no reviewer can
> safely decide their piece without the whole picture.
>
> **Recommended direction (veto or adjust):** adopt pixi as the dev-env + lock
> solution, decline conda-lock as redundant once pixi lands, and evaluate the
> flit backend swap separately since it's independent. Comment that direction
> on a new `packaging strategy` decision issue and link all three.
>
> **Reversible and small.** You're not merging anything today — you're stating
> a direction so three contributors can move, and it can still be revised
> before any merge. Cost of not doing it: contributors idle up to ~3 years, and
> every future packaging PR inherits the same paralysis.

## Tone

Direct, concrete, no filler. The maintainer reads this three times a week; it
must stay skimmable. The FYI zone is meant to be skipped — do not editorialize
it.
