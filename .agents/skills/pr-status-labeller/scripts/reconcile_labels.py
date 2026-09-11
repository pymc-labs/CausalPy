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
"""Reconcile `status:*` labels on open CausalPy PRs.

Deterministic. Calls the shared triage core, then for each PR adds/removes
labels so the live `status:*` set matches computed reality. Touches ONLY the
`status:*` namespace — never `review:*`, `major`, `needs:maintainer-decision`,
or any other semantic label a human owns.

Dry-run by default. Pass --apply to actually mutate GitHub.

Usage:
    python reconcile_labels.py [--repo OWNER/NAME] [--apply]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import subprocess
import sys

_CORE = os.path.join(
    os.path.dirname(__file__), "..", "..", "_shared", "pr_triage", "labels.py"
)
_spec = importlib.util.spec_from_file_location("pr_triage_labels", _CORE)
core = importlib.util.module_from_spec(_spec)
sys.modules["pr_triage_labels"] = core  # needed so dataclass annotations resolve
_spec.loader.exec_module(core)

# Colour + description for each managed label, created on first apply if absent.
LABEL_DEFS = {
    "status:conflicting": ("B60205", "Has merge conflicts with the base branch"),
    "status:ci-failing": ("B60205", "Remote CI is red"),
    "status:waiting-on-author": ("FBCA04", "Changes requested; awaiting author"),
    "status:ready-for-review": ("0E8A16", "Clean and green; awaiting a reviewer"),
    "status:aging": ("D4C5F9", f"No activity in >={core.AGING_DAYS}d"),
    "status:stale": ("5319E7", f"No activity in >={core.STALE_DAYS}d"),
}

# LABEL_DEFS must cover exactly the managed namespace declared in core.
assert set(LABEL_DEFS) == core.STATUS_LABELS, (
    f"LABEL_DEFS out of sync with STATUS_LABELS: {set(LABEL_DEFS) ^ core.STATUS_LABELS}"
)


def _gh(args: list[str]) -> None:
    subprocess.run(["gh", *args], check=True, capture_output=True, text=True)


def _label_args(flag: str, labels: set[str]) -> list[str]:
    """Flatten a set of labels into repeated `--flag name` CLI args."""
    out: list[str] = []
    for name in labels:
        out += [flag, name]
    return out


def ensure_labels(repo: str) -> None:
    for name, (color, desc) in LABEL_DEFS.items():
        with contextlib.suppress(subprocess.CalledProcessError):
            _gh(
                [
                    "label",
                    "create",
                    name,
                    "--repo",
                    repo,
                    "--color",
                    color,
                    "--description",
                    desc,
                    "--force",
                ]
            )


def current_status_labels(pr: dict) -> set[str]:
    return {
        lab["name"] for lab in pr.get("labels", []) if lab["name"].startswith("status:")
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=core.DEFAULT_REPO)
    ap.add_argument(
        "--apply", action="store_true", help="mutate GitHub (default: dry-run)"
    )
    args = ap.parse_args()
    core.REPO = args.repo

    raw = core.fetch_prs(args.repo)
    facts = {f.number: f for f in core.classify(raw)}

    if args.apply:
        ensure_labels(args.repo)

    changes = 0
    for pr in raw:
        n = pr["number"]
        have = current_status_labels(pr)
        want = set(facts[n].status_labels)
        add, remove = want - have, have - want
        if not add and not remove:
            continue
        changes += 1
        verb = "APPLY" if args.apply else "DRY  "
        print(f"[{verb}] #{n}: +{sorted(add) or '-'}  -{sorted(remove) or '-'}")
        if args.apply:
            if add:
                _gh(
                    [
                        "pr",
                        "edit",
                        str(n),
                        "--repo",
                        args.repo,
                        *_label_args("--add-label", add),
                    ]
                )
            if remove:
                _gh(
                    [
                        "pr",
                        "edit",
                        str(n),
                        "--repo",
                        args.repo,
                        *_label_args("--remove-label", remove),
                    ]
                )

    mode = "applied" if args.apply else "dry-run (pass --apply to write)"
    print(f"\n{changes} PR(s) with label changes — {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
