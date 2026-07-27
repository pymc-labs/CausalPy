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
"""Build the Mon/Wed/Fri CausalPy PR digest from the deterministic triage core.

This produces the *structure and data* of the digest deterministically. It
selects exactly ONE hard item (confirmed ceiling: maintainers realistically do
no more than one deep-focus decision per day) and groups everything else. The
`pr-digest` skill then adds the light judgment layer: the recommended-default
and reversible-fork framing for the single hard item, and any cross-PR
clustering insight (e.g. "these three are one packaging decision").

Output is a JSON object with a ready-to-post `markdown` field plus the
`hard_item` payload the agent should flesh out before posting.

Usage:
    python build_digest.py [--repo OWNER/NAME] [--day Mon] [--mention <id>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

_CORE = os.path.join(
    os.path.dirname(__file__), "..", "..", "_shared", "pr_triage", "labels.py"
)
_spec = importlib.util.spec_from_file_location("pr_triage_labels", _CORE)
core = importlib.util.module_from_spec(_spec)
sys.modules["pr_triage_labels"] = core  # needed so dataclass annotations resolve
_spec.loader.exec_module(core)


def _fmt(f: dict) -> str:
    return f"[#{f['number']} {f['title'][:60]}]({f['url']})"


def _link(f: dict) -> str:
    return f"[#{f['number']}]({f['url']})"


def pick_hard_item(prs: list[dict]) -> dict | None:
    """The single deep-focus item. Candidates are decision-needed PRs and
    major PRs that are ready for review. Rank by cost-of-waiting (age), so the
    longest-frozen decision leads. Returns None if nothing qualifies."""
    candidates = [
        f
        for f in prs
        if f["decision_needed"]
        or (f["major"] and f["next_action"] == "ready-for-review")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda f: f["age_days"])


def build(payload: dict, day: str, mention: str | None) -> dict:
    prs = payload["prs"]
    by = lambda a: [f for f in prs if f["next_action"] == a]

    hard = pick_hard_item(prs)
    other_hard = [
        f
        for f in prs
        if (
            f["decision_needed"]
            or (f["major"] and f["next_action"] == "ready-for-review")
        )
        and (hard is None or f["number"] != hard["number"])
    ]

    reviews = sorted(
        by("ready-for-review"),
        key=lambda f: (f["author_class"] != "external", f["idle_days"]),
    )
    waiting = by("waiting-on-author")
    mechanical = by("mechanical")
    stale_drafts = by("stale-draft") + by("aging-draft")
    in_flight = by("in-flight-draft")
    to_merge = by("ready-to-merge")

    m: list[str] = []
    tag = f" <@{mention}>" if mention else ""
    m.append(f"**CausalPy PR digest — {day}**{tag}")
    m.append(
        f"{payload['open_count']} open. **Today: "
        f"{'1 decision + ' if hard else ''}{len(reviews)} quick reviews"
        f"{f' + {len(to_merge)} ready to merge' if to_merge else ''}.** "
        "Everything below the line is tracked and needs nothing from you. "
        "This digest never acts on its own — it only recommends."
    )

    if hard:
        m.append("\n**🎯 Your one decision today** (~20 min, needs real focus)")
        m.append("\n{{HARD_ITEM_FRAMING}}")  # agent fills: context, default, cost
        m.append(
            f"\nLead PR: {_fmt(hard)} ({hard['author']}, "
            f"open {hard['age_days']}d, idle {hard['idle_days']}d)."
        )
        if other_hard:
            queued = ", ".join(_link(f) for f in other_hard[:4])
            m.append(
                f"\n*{len(other_hard)} more decision(s) queued behind this one "
                f"({queued}). Deliberately not asking today — one hard thing "
                "per digest.*"
            )

    if to_merge:
        m.append("\n**🟢 Approved, just needs a merge click**")
        for f in to_merge:
            m.append(f"- {_fmt(f)} ({f['author']})")

    if reviews:
        m.append(
            "\n**✅ Quick reviews — clean reads, no yak-shaving** "
            "(do as many as you have time for, external contributors first)"
        )
        for f in reviews:
            risk = f" `review:{f['risk']}`" if f["risk"] else ""
            ext = " — external" if f["author_class"] == "external" else ""
            m.append(f"- {_fmt(f)} ({f['author']}{ext}){risk}")

    m.append("\n━━━━━━━━━━━━━━━━━━━━")
    m.append("**Below: tracked, no action needed. Skim or skip.**")

    if waiting:
        m.append(
            f"\n**⏳ Waiting on authors ({len(waiting)})** — changes "
            "requested, ball in their court: "
            + ", ".join(_link(f) for f in waiting)
            + "."
        )
    if mechanical:
        m.append(
            f"\n**🔧 Mechanical, auto-nudged once Tier 2 is on "
            f"({len(mechanical)})** — conflicts or CI red, no judgment "
            "needed: "
            + ", ".join(_link(f) for f in mechanical[:8])
            + ("…" if len(mechanical) > 8 else "")
            + "."
        )
    if stale_drafts:
        oldest = max(stale_drafts, key=lambda f: f["age_days"])
        m.append(
            f"\n**🗂️ Stale/aging drafts — a nudge, not a task "
            f"({len(stale_drafts)})** — oldest is {_link(oldest)} "
            f"(opened {oldest['age_days']}d ago, idle "
            f"{oldest['idle_days']}d). Worth a bulk abandon-or-resume "
            "pass some quiet afternoon."
        )
    if in_flight:
        m.append(
            f"\n**🌱 {len(in_flight)} active drafts in flight** — WIP, "
            "correctly silent. A draft only surfaces above once it ages "
            f"past {core.AGING_DAYS}d idle."
        )

    return {
        "day": day,
        "markdown": "\n".join(m),
        "hard_item": hard,
        "other_hard": other_hard,
        "counts": {
            "open": payload["open_count"],
            "reviews": len(reviews),
            "waiting": len(waiting),
            "mechanical": len(mechanical),
            "stale_drafts": len(stale_drafts),
            "in_flight": len(in_flight),
            "to_merge": len(to_merge),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=core.DEFAULT_REPO)
    ap.add_argument("--day", default="Today")
    ap.add_argument("--mention", default=None, help="Discord user id to tag")
    args = ap.parse_args()
    core.REPO = args.repo
    facts = core.classify(core.fetch_prs(args.repo))
    payload = {
        "repo": args.repo,
        "open_count": len(facts),
        "prs": [core.asdict(f) for f in facts],
    }
    result = build(payload, args.day, args.mention)
    json.dump(result, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
