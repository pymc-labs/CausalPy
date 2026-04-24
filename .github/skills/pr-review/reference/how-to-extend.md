# How to Extend This Skill

This skill should grow with the codebase. New patterns get added when they recur; old patterns get pruned when they get formalised into hooks.

## When to add a new pattern

Add a pattern when **either** is true:

- You've seen the same shape of issue in **two or more PRs** (it's not a one-off).
- The pattern is specific enough to CausalPy that a generic reviewer wouldn't catch it (it's not common-knowledge linting).

Don't add a pattern just because it came up once. The risk of skill bloat is real — a 200-line checklist is worse than a 50-line one because agents start cherry-picking.

## Where to add it

Choose the smallest scope that fits:

| Pattern type | Where to add |
|---|---|
| Severity-sorted "look for X" | [what-to-look-for.md](what-to-look-for.md) under the right severity heading |
| CausalPy source-code convention | [code-patterns.md](code-patterns.md) |
| Notebook / docs convention | [docs-patterns.md](docs-patterns.md) |
| Comment-shape template | [posting-comments.md](posting-comments.md) |
| Workflow / process change | [workflow.md](workflow.md) |

If a new pattern doesn't fit any existing file, that's a signal the skill is growing a new dimension and may need a new reference file. Don't force-fit.

## Required structure for a new pattern

Each pattern needs all four:

1. **A short ID** (e.g. `MF-1`, `SF-3`, `N-7`) for cross-referencing.
2. **A one-line title** that says what the pattern is.
3. **A canonical example**: a real PR (e.g. "PR #826") and a specific instance, so future readers can ground the abstraction in a concrete case.
4. **A "how to spot" prompt**: what does the agent actually look at to decide if this pattern is present? Vague patterns produce vague flags.

Template:

```markdown
### <ID>: <One-line title>

<2–4 sentences explaining what the pattern is and why it matters.>

- **Canonical example**: <PR ref + specific instance>.
- **How to spot**: <concrete diagnostic prompt>.
- <Optional: links to related patterns or code-patterns.md sections>.
```

## When to prune

Prune a pattern when:

- A hook or CI check now enforces it mechanically. The agent shouldn't waste attention on what tooling already catches.
- The pattern hasn't recurred in 12 months and the originating context has changed.
- A new pattern subsumes it. Merge the two; don't leave both.

Pruning is part of maintenance, not vandalism. The skill is more useful at 200 well-curated lines than 600 stale ones.

## When to formalise into a hook

If a pattern in `what-to-look-for.md` has a **mechanical answer** — same input always produces the same correct verdict — it should probably become a hook, not stay an agent task.

Mechanical patterns from past reviews that were (or should be) formalised:

- **Single H1 per notebook** → issue #863, formalised in #866 (`validate-notebooks` pre-commit hook); pattern pruned from `docs-patterns.md` at the same time.
- **Notebook orphaned from `toctree`** → under discussion.
- **Constrained string param not `Literal`** → potential Ruff/mypy rule (deferred).

When you formalise a pattern, **remove it from this skill** at the same time. Don't leave both — divergence is worse than either alone.

## Six-monthly review

On a recurring cadence (every six months or after every ~10 PRs reviewed), do a quick pass:

1. Are any patterns stale or unused?
2. Are any patterns now hook-enforced and need pruning?
3. Have new patterns recurred enough to add?
4. Is `SKILL.md` still under 500 lines / 5000 tokens?

If yes to any of (1) (2) (3), update accordingly. If no to (4), refactor — split a reference file, move detail out of `SKILL.md`.

## Worked example: how this skill itself was created

Initial creation drew on PR #826 review feedback as the canonical seed:

- 2 must-fix items → MF-1 (`_clone()` priors), MF-2 (`Literal` for constrained strings).
- 6 should-fix items → SF-1 through SF-6.
- 7 nits → N-1 through N-7 (a few omitted as too one-off).
- 3 process patterns → P-1 (mega payload), P-2 (superseded commits), P-3 (orphan notebook).

Each pattern was checked against the four-element template (ID, title, example, how-to-spot) before inclusion. Patterns that were just "a single judgement call" without a recognisable shape were kept out — they're the kind of thing agents naturally do well anyway and don't benefit from a checklist entry.
