#!/usr/bin/env python3
#   Copyright 2026 - 2026 The PyMC Labs Developers
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
#!/usr/bin/env python3
"""
Generate comprehensive AI-style summaries for notifications.

This creates detailed, actionable summaries that help users decide
which notifications to prioritize.
"""

import json

# Known bots
BOT_USERNAMES = {
    "codecov",
    "codecov[bot]",
    "codecov-commenter",
    "pre-commit-ci",
    "pre-commit-ci[bot]",
    "dependabot",
    "dependabot[bot]",
    "github-actions",
    "github-actions[bot]",
    "renovate",
    "renovate[bot]",
    "review-notebook-app[bot]",
    "review-notebook-app",
}


def generate_summary(n):
    """Generate a comprehensive summary for a notification."""
    reason = n.get("reason", "")
    category = n.get("category", "")
    title = n.get("title", "")
    ntype = n.get("type", "")
    details = n.get("details", {})
    tags = [t[1] for t in n.get("tags", [])]

    body = details.get("body", "") or ""
    comments = details.get("comments", [])
    reviews = details.get("reviews", [])

    # Filter to human comments only
    human_comments = [
        c for c in comments if c.get("author", {}).get("login", "") not in BOT_USERNAMES
    ]
    human_reviews = [
        r
        for r in reviews
        if r.get("author", {}).get("login", "") not in BOT_USERNAMES and r.get("body")
    ]

    summary_parts = []

    # === RESOLVED ITEMS ===
    if category == "resolved":
        state = details.get("state", "")
        if details.get("mergedAt"):
            summary_parts.append("**Status:** ✅ Merged")
            summary_parts.append("")
            summary_parts.append(
                "This PR has been successfully merged. No further action required."
            )
            if body:
                summary_parts.append("")
                summary_parts.append("**What it did:**")
                body_preview = body[:300].replace("\n", " ").strip()
                summary_parts.append(f"> {body_preview}...")
        elif state in ["CLOSED", "closed"]:
            summary_parts.append("**Status:** ⛔ Closed")
            summary_parts.append("")
            summary_parts.append(
                f"This {ntype.lower()} was closed without merging. No action needed."
            )
        else:
            summary_parts.append("**Status:** ✅ Resolved")
        return "\n".join(summary_parts)

    # === HIDDEN ITEMS ===
    if category == "hidden":
        return "**Status:** 🙈 CI/Workflow Activity\n\nThis is automated CI activity. Usually no action needed unless investigating build failures."

    # === ACTION ITEMS ===
    if reason == "mention":
        summary_parts.append("**🔔 You were mentioned**")
        summary_parts.append("")

        if human_comments:
            last = human_comments[-1]
            author = last.get("author", {}).get("login", "Someone")
            comment_body = last.get("body", "")
            summary_parts.append(f"**{author}** said:")
            summary_parts.append(f"> {comment_body[:500]}")
            summary_parts.append("")

            # Check if it's a question
            if "?" in comment_body or "what do you think" in comment_body.lower():
                summary_parts.append(
                    "**Action Required:** This appears to be a question directed at you. Please review and respond."
                )
            else:
                summary_parts.append(
                    "**Action Required:** You were mentioned - review the context and respond if needed."
                )

        elif human_reviews:
            last = human_reviews[-1]
            author = last.get("author", {}).get("login", "Someone")
            review_body = last.get("body", "")
            summary_parts.append(f"**{author}** mentioned you in a review:")
            summary_parts.append(f"> {review_body[:500]}")
            summary_parts.append("")
            summary_parts.append(
                "**Action Required:** Review the feedback and respond."
            )
        else:
            summary_parts.append("You were mentioned in this discussion.")
            summary_parts.append("")
            summary_parts.append(
                "**Action Required:** Check the thread for context and respond if needed."
            )

    elif reason == "review_requested":
        summary_parts.append("**📋 Review Requested**")
        summary_parts.append("")

        if body:
            # Extract key info from PR body
            body_lines = body.split("\n")
            body_preview = "\n".join(body_lines[:10])
            summary_parts.append("**What this PR does:**")
            summary_parts.append(f"> {body_preview[:600]}")
            summary_parts.append("")

        # Check CI status
        if "CI FAILED" in tags:
            summary_parts.append(
                "⚠️ **CI is failing** - you may want to wait for fixes before reviewing."
            )
            summary_parts.append("")

        if "DRAFT PR" in tags:
            summary_parts.append(
                "📝 **This is a draft PR** - the author may still be working on it."
            )
            summary_parts.append("")

        summary_parts.append(
            "**Action Required:** Review this PR and provide feedback (approve, request changes, or comment)."
        )

    elif reason == "author":
        summary_parts.append(f"**📣 Activity on your {ntype}**")
        summary_parts.append("")

        # Check for changes requested
        if "CHANGES REQUESTED" in tags:
            summary_parts.append("**🔶 Changes Requested**")
            summary_parts.append("")
            for r in human_reviews[-3:]:
                if r.get("state") == "CHANGES_REQUESTED":
                    author = r.get("author", {}).get("login", "")
                    review_body = r.get("body", "")
                    if review_body:
                        summary_parts.append(f"**{author}** requested changes:")
                        summary_parts.append(f"> {review_body[:400]}")
                        summary_parts.append("")
            summary_parts.append(
                "**Action Required:** Address the requested changes and update the PR."
            )

        elif "PR APPROVED" in tags:
            summary_parts.append("**✅ Your PR has been approved!**")
            summary_parts.append("")
            approvers = [
                r.get("author", {}).get("login", "")
                for r in reviews
                if r.get("state") == "APPROVED"
            ]
            if approvers:
                summary_parts.append(f"Approved by: {', '.join(approvers)}")
            summary_parts.append("")
            summary_parts.append("**Action:** Merge when ready (if CI is passing).")

        elif "CI FAILED" in tags:
            summary_parts.append("**🔴 CI Failed**")
            summary_parts.append("")
            summary_parts.append("Your PR has failing CI checks.")
            summary_parts.append("")
            summary_parts.append(
                "**Action Required:** Check the CI logs and fix the failing tests/checks."
            )

        elif human_comments or human_reviews:
            summary_parts.append("**💬 New Feedback**")
            summary_parts.append("")

            # Show recent human activity
            all_activity = []
            for c in human_comments[-3:]:
                all_activity.append(("comment", c))
            for r in human_reviews[-2:]:
                if r.get("body"):
                    all_activity.append(("review", r))

            for atype, item in all_activity[-3:]:
                author = item.get("author", {}).get("login", "Someone")
                body_text = item.get("body", "")[:300]
                if atype == "review":
                    state = item.get("state", "")
                    state_emoji = {"APPROVED": "✅", "CHANGES_REQUESTED": "🔶"}.get(
                        state, "💬"
                    )
                    summary_parts.append(f"**{author}** ({state_emoji} review):")
                else:
                    summary_parts.append(f"**{author}** commented:")
                summary_parts.append(f"> {body_text}")
                summary_parts.append("")

            summary_parts.append(
                "**Action:** Review the feedback and respond if needed."
            )

        else:
            summary_parts.append(
                f"There's activity on your {ntype.lower()}, but no specific feedback to highlight."
            )

    elif reason == "subscribed":
        summary_parts.append("**👀 Watching**")
        summary_parts.append("")
        if body:
            body_preview = body[:400].replace("\n", " ")
            summary_parts.append(f"> {body_preview}")
            summary_parts.append("")
        summary_parts.append(
            "**FYI only** - you're watching this thread. No action required unless you want to participate."
        )

    else:
        # Generic notification
        if body:
            body_preview = body[:500].replace("\n", " ")
            summary_parts.append(body_preview)
        else:
            summary_parts.append(f"Notification about: {title}")

    return "\n".join(summary_parts) if summary_parts else f"Notification about: {title}"


def main():
    from pathlib import Path

    # Get the project root (parent of .cursor/commands)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    scratch_dir = project_root / ".scratch"

    # Load enriched notifications
    input_file = scratch_dir / "notifications-enriched.json"
    with open(input_file) as f:
        notifications = json.load(f)

    print(
        f"Generating comprehensive summaries for {len(notifications)} notifications..."
    )

    # Generate summaries
    for n in notifications:
        n["summary"] = generate_summary(n)
        preview = n["summary"][:60].replace("\n", " ")
        print(f"  #{n['number']}: {preview}...")

    # Save with summaries
    output_file = scratch_dir / "notifications-with-summaries.json"
    with open(output_file, "w") as f:
        json.dump(notifications, f, indent=2)

    print(f"\n✅ Saved to {output_file}")


if __name__ == "__main__":
    main()
