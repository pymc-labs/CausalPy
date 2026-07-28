#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Deterministic PR triage core for CausalPy.

Single source of truth for "what are the facts about each open PR". Both the
`pr-status-labeller` skill (writes `status:*` labels to GitHub) and the
`pr-digest` skill (Mon/Wed/Fri Discord digest) call this module so the two
never disagree. No LLM judgment lives here: everything is a pure function of
the GitHub API. The only judgment layer is the prose/framing the digest agent
adds on top of this structured output.

Usage:
    python labels.py [--repo OWNER/NAME] [--json]

Requires the `gh` CLI authenticated with read access to the repo.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field

# --- Configuration (edit here, nowhere else) --------------------------------

DEFAULT_REPO = "pymc-labs/CausalPy"

# Author classification. Everyone not listed and not a bot is "external".
MAINTAINERS = {"drbenvincent"}
LABS = {
    "NathanielF",
    "juanitorduz",
    "cetagostini",
    "twiecki",
    "ricardoV94",
    "OriolAbril",
    "daimon-pymclabs",  # labs automation account (still bot-like, see BOT rule)
}

# Idle-band thresholds, in days. Confirmed with maintainer:
#   nudge (aging) at 30d, stale warning at 90d.
AGING_DAYS = 30
STALE_DAYS = 90

# Precedence order for the single derived `next_action`. First match wins:
# `_next_action` walks this list and returns the first action whose predicate
# holds, and the CLI table / consumers sort by it. This is the ONE place
# priority judgment is encoded; tune it here. `ready-for-review` is the
# always-true fallback and must stay last.
NEXT_ACTION_ORDER = [
    "decision",  # needs a maintainer decision (the hard queue)
    "stale-draft",  # your own draft, idle >= STALE_DAYS
    "aging-draft",  # your own draft, idle >= AGING_DAYS
    "in-flight-draft",  # fresh draft, WIP, silent
    "waiting-on-author",  # changes requested, ball in author's court
    "ready-to-merge",  # approved + clean + not red
    "mechanical",  # conflicting or CI red (agent-fixable in Tier 2)
    "ready-for-review",  # clean, green/pending, awaiting a reviewer
]

# Rank lookup so the CLI table and any consumer order by this list, not
# alphabetically. Editing NEXT_ACTION_ORDER above now actually reorders output.
NEXT_ACTION_RANK = {a: i for i, a in enumerate(NEXT_ACTION_ORDER)}

# The `status:*` labels the labeller is allowed to manage. It touches ONLY this
# namespace, never `review:*`, `major`, `needs:maintainer-decision`, etc.
STATUS_LABELS = {
    "status:conflicting",
    "status:ci-failing",
    "status:waiting-on-author",
    "status:ready-for-review",
    "status:aging",
    "status:stale",
}

# -----------------------------------------------------------------------------


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _parse_ts(ts: str) -> _dt.datetime:
    return _dt.datetime.fromisoformat(ts)


def _days_since(ts: str) -> int:
    return (_now() - _parse_ts(ts)).days


def _last_activity(pr: dict) -> str:
    """Timestamp of the PR's last real activity, used to measure idleness.

    Intentionally NOT `updatedAt`: that field is bumped by every mutation to
    the PR, including the label-only `gh pr edit` writes the labeller itself
    performs. If idleness were read from `updatedAt`, applying `status:aging`
    would reset the clock, the next reconcile run would see a "fresh" PR and
    remove the label it just added, and aging/stale state could never stick.

    Instead we take the most recent of: PR creation, last commit, and last
    human/CI comment. None of those are moved by a label edit, so the idle
    figure is stable across labeller runs, and the digest (which shares this
    core) reads the same number."""
    stamps = [pr["createdAt"]]
    for commit in pr.get("commits") or []:
        # committedDate is the push time; take them all so a rebase that
        # reorders commits can't hide the newest one.
        if commit.get("committedDate"):
            stamps.append(commit["committedDate"])
    for c in pr.get("comments") or []:
        if c.get("createdAt"):
            stamps.append(c["createdAt"])
    return max(stamps, key=_parse_ts)


def _gh_json(args: list[str]) -> object:
    out = subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out) if out.strip() else None


def fetch_prs(repo: str) -> list[dict]:
    """Fetch open PRs. Forces GitHub to compute the lazily-evaluated
    `mergeable` field, which comes back UNKNOWN in bulk listings.

    We deliberately fetch `comments` (and `commits`, per-PR below) rather than
    trusting `updatedAt` to measure idleness. `updatedAt` is bumped by ANY edit
    to the PR, including the label-only `gh pr edit` writes this very system
    makes when it applies `status:aging`/`status:stale`. Reading idleness from
    it would mean each labeller apply resets the idle clock, so the next run
    sees a fresh PR and strips the aging/stale label it just added: the state
    could never persist. See `_last_activity`.

    `commits` is fetched per-PR, NOT in the bulk list: a bulk `pr list` that
    includes `commits` (each carrying an authors connection) blows past
    GitHub's 500k GraphQL node budget. We piggy-back the commit fetch onto the
    same per-PR `gh pr view` used to resolve lazy `mergeable`."""
    fields = (
        "number,title,author,isDraft,mergeable,reviewDecision,"
        "updatedAt,createdAt,labels,statusCheckRollup,comments"
    )
    prs = (
        _gh_json(
            [
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "open",
                "--limit",
                "200",
                "--json",
                fields,
            ]
        )
        or []
    )
    # Per-PR pass: fetch `commits` (too expensive in bulk, see docstring) and,
    # in the same call, resolve any UNKNOWN `mergeable` (`gh pr view` triggers
    # the lazy computation). One `gh pr view` per PR covers both.
    for pr in prs:
        want = ["commits"]
        if pr.get("mergeable") == "UNKNOWN":
            want.append("mergeable")
        with contextlib.suppress(subprocess.CalledProcessError):
            v = _gh_json(
                [
                    "pr",
                    "view",
                    str(pr["number"]),
                    "--repo",
                    repo,
                    "--json",
                    ",".join(want),
                ]
            )
            if v:
                pr["commits"] = v.get("commits", [])
                if "mergeable" in want:
                    pr["mergeable"] = v.get("mergeable", "UNKNOWN")
    return prs


def _ci_state(pr: dict) -> str:
    rollup = pr.get("statusCheckRollup") or []
    states = [c.get("conclusion") or c.get("state") for c in rollup]
    if not states:
        return "none"
    bad = {"FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED"}
    if any(s in bad for s in states):
        return "red"
    pending = {"PENDING", "IN_PROGRESS", "QUEUED", "EXPECTED", None}
    if any(s in pending for s in states):
        return "pending"
    return "green"


def _author_class(login: str) -> str:
    if login in MAINTAINERS:
        return "maintainer"
    if "[bot]" in login or login.startswith("app/") or login.endswith("-pymclabs"):
        return "bot"
    if login in LABS:
        return "labs"
    return "external"


def _idle_band(idle: int) -> str:
    if idle >= STALE_DAYS:
        return "stale"
    if idle >= AGING_DAYS:
        return "aging"
    return "fresh"


@dataclass
class PRFacts:
    number: int
    title: str
    author: str
    url: str
    author_class: str
    lifecycle: str  # draft | ready
    conflict: str  # clean | conflicting | unknown
    ci: str  # green | red | pending | none
    review: str  # none | required | changes-requested | approved
    decision_needed: bool
    major: bool
    risk: str | None  # high | medium | low | None
    idle_days: int
    age_days: int
    idle_band: str  # fresh | aging | stale
    next_action: str
    status_labels: list[str] = field(default_factory=list)


def _review_state(reviewDecision: str | None) -> str:
    return {
        None: "none",
        "REVIEW_REQUIRED": "required",
        "CHANGES_REQUESTED": "changes-requested",
        "APPROVED": "approved",
    }.get(reviewDecision, "none")


def _next_action(f: dict) -> str:
    """Derive the single `next_action`. Precedence is `NEXT_ACTION_ORDER`,
    read verbatim: we walk that list and return the first action whose
    predicate matches, so reordering the list up top genuinely reorders the
    classification (true first-match-wins). `ready-for-review` is the
    always-true fallback and must stay last in the list."""
    draft = f["lifecycle"] == "draft"
    predicates = {
        "decision": lambda: f["decision_needed"],
        "stale-draft": lambda: draft and f["idle_band"] == "stale",
        "aging-draft": lambda: draft and f["idle_band"] == "aging",
        "in-flight-draft": lambda: draft,
        "waiting-on-author": lambda: f["review"] == "changes-requested",
        "ready-to-merge": lambda: (
            f["review"] == "approved" and f["conflict"] == "clean" and f["ci"] != "red"
        ),
        "mechanical": lambda: f["conflict"] == "conflicting" or f["ci"] == "red",
        "ready-for-review": lambda: True,
    }
    # Guard against drift between the precedence list and the predicate map.
    assert set(predicates) == set(NEXT_ACTION_ORDER), (
        "predicate keys out of sync with NEXT_ACTION_ORDER: "
        f"{set(predicates) ^ set(NEXT_ACTION_ORDER)}"
    )
    for action in NEXT_ACTION_ORDER:
        if predicates[action]():
            return action
    return "ready-for-review"


def _status_labels_for(f: dict) -> list[str]:
    """Derive the `status:*` label set. Drafts stay out of the reviewer-facing
    statuses; they only carry aging/stale flags."""
    labels: list[str] = []
    draft = f["lifecycle"] == "draft"
    if f["idle_band"] == "stale":
        labels.append("status:stale")
    elif f["idle_band"] == "aging":
        labels.append("status:aging")
    if not draft:
        if f["conflict"] == "conflicting":
            labels.append("status:conflicting")
        if f["ci"] == "red":
            labels.append("status:ci-failing")
        if f["review"] == "changes-requested":
            labels.append("status:waiting-on-author")
        if f["next_action"] == "ready-for-review":
            labels.append("status:ready-for-review")
    # Guard against drift: the labeller only manages the STATUS_LABELS set.
    assert set(labels) <= STATUS_LABELS, (
        f"emitted labels {set(labels) - STATUS_LABELS} are outside STATUS_LABELS"
    )
    return labels


def classify(prs: list[dict]) -> list[PRFacts]:
    results: list[PRFacts] = []
    for pr in prs:
        login = pr["author"]["login"]
        names = [lab["name"] for lab in pr.get("labels", [])]
        risk = next(
            (n.split(":", 1)[1] for n in names if n.startswith("review:")), None
        )
        merge = pr.get("mergeable", "UNKNOWN")
        conflict = {"CONFLICTING": "conflicting", "MERGEABLE": "clean"}.get(
            merge, "unknown"
        )
        f = {
            "number": pr["number"],
            "title": pr["title"],
            "author": login,
            "url": f"https://github.com/{REPO}/pull/{pr['number']}",
            "author_class": _author_class(login),
            "lifecycle": "draft" if pr["isDraft"] else "ready",
            "conflict": conflict,
            "ci": _ci_state(pr),
            "review": _review_state(pr.get("reviewDecision")),
            "decision_needed": "needs:maintainer-decision" in names,
            "major": "major" in names,
            "risk": risk,
            "idle_days": _days_since(_last_activity(pr)),
            "age_days": _days_since(pr["createdAt"]),
        }
        f["idle_band"] = _idle_band(f["idle_days"])
        f["next_action"] = _next_action(f)
        f["status_labels"] = _status_labels_for(f)
        results.append(PRFacts(**f))
    return results


REPO = DEFAULT_REPO


def main() -> int:
    global REPO
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--json", action="store_true", help="emit JSON to stdout")
    args = ap.parse_args()
    REPO = args.repo
    facts = classify(fetch_prs(REPO))
    payload = {
        "repo": REPO,
        "generated_at": _now().isoformat(),
        "open_count": len(facts),
        "prs": [asdict(f) for f in facts],
    }
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        for f in sorted(
            facts,
            key=lambda x: (
                NEXT_ACTION_RANK.get(x.next_action, len(NEXT_ACTION_ORDER)),
                -x.idle_days,
            ),
        ):
            print(
                f"#{f.number:<5} {f.next_action:<18} idle{f.idle_days:>4}d "
                f"{f.author_class:<10} {f.title[:55]}"
            )
        print(f"\n{len(facts)} open PRs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
