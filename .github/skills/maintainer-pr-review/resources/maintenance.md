# Skill Maintenance

Use this resource when updating `maintainer-pr-review` with patterns learned from future reviews.

## When to Add a Pattern

Add a pattern when either condition is true:

- The same issue shape has appeared in two or more PRs.
- The pattern is CausalPy-specific enough that a generic reviewer is unlikely to catch it.

Do not add a pattern just because it appeared once. Skill bloat makes agents skim and weakens review quality.

## Where to Add It

Choose the smallest useful scope:

| Pattern type | File |
|---|---|
| PR-type checklist | The matching PR-type resource file |
| Severity-sorted recurring pattern | `review-patterns.md` |
| CausalPy source-code convention | `code-patterns.md` |
| Docs or notebook convention | `docs-patterns.md` |
| Review workflow or claim verification | `workflow.md` |
| Comment template or posting rule | `review-comments.md` |

If a pattern does not fit any file, consider whether it is truly recurring before adding a new resource.

## Pattern Template

Each recurring pattern should include:

1. A short ID if it belongs in `review-patterns.md`, such as `MF-1`, `SF-3`, or `N-7`.
2. A one-line title.
3. A concrete example or review context, preferably with a PR reference.
4. A "how to spot" prompt that tells the agent exactly what to inspect.

```markdown
### <ID>: <One-line title>

<Two to four sentences explaining the pattern and why it matters.>

- Canonical example: <PR reference and concrete instance>.
- How to spot: <specific diagnostic prompt>.
```

## When to Prune

Prune a pattern when:

- A hook or CI check now enforces it mechanically.
- The pattern has not recurred in a long time and the originating context no longer matters.
- A newer pattern subsumes it.

When a pattern becomes a hook, remove it from the skill at the same time. Keeping both creates drift.

## Owner and Trigger

The owner is the maintainer or agent using this skill when they notice drift, repeated review misses, or a pattern that has become mechanical enough for automation. When that happens, create a GitHub issue against the project with the observed pattern, example PRs, and the proposed resource file to update.

Use these triggers:

- A pattern recurs in two or more PRs.
- A review finds that the skill sent an agent to a nonexistent or local-only dependency.
- A hook or CI check now covers a pattern documented here.
- Roughly ten PR reviews have used this skill since the last maintenance pass.
- Six months have passed since the last maintenance pass.

## Validation for Skill Changes

For meaningful changes to this skill, run formatting checks and do at least one real-world smoke review when feasible. A smoke review can be a focused review of an open PR, an existing review comment thread, or a recently merged PR; record what was reviewed and what the skill helped catch in the PR description.

## Review Cadence

Every six months or after roughly ten PR reviews, check:

1. Are any patterns stale?
2. Are any patterns now hook-enforced?
3. Have new patterns recurred enough to add?
4. Is `SKILL.md` still short enough to act as a router?
5. Are resource links still one level deep and easy to discover?
