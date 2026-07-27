# Shared PR triage core

`labels.py` is the single deterministic source of truth for CausalPy PR triage.
It fetches open PRs via `gh`, computes a set of orthogonal facts per PR
(lifecycle, conflict, CI, review, decision-needed, risk, idle-band,
author-class), and derives one `next_action` per PR via an explicit precedence
order.

Two skills consume it, and consume the *same* function so they can never
disagree:

- `pr-status-labeller` — writes `status:*` labels to GitHub.
- `pr-digest` — produces the read-only Mon/Wed/Fri Discord digest.

No LLM judgment lives here. All configuration (maintainer/labs handles, idle
thresholds, precedence order, managed label set) is at the top of `labels.py`;
edit it there and nowhere else.

Run standalone to inspect classification:

```bash
python labels.py                # human-readable table
python labels.py --json         # structured output
```
