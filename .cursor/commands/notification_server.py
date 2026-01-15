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
Local web server for interactive GitHub notification management.

Serves the notification summary as an HTML page with "Mark Done" buttons
that call the GitHub API to archive notifications.

Usage:
    python .cursor/commands/notification_server.py [--port 8765] [--no-open]

Prerequisites:
    Run fetch_notifications.py first to create the enriched JSON file.
    Optionally, have the Cursor agent add AI-generated summaries.

The server will:
1. Load notifications from .scratch/notifications-with-summaries.json
   (or .scratch/notifications-enriched.json as fallback)
2. Serve an interactive HTML page with tabs for each category
3. Handle "Mark Done" API calls via DELETE /notifications/threads/{id}
"""

import argparse
import json
import subprocess
import sys
import webbrowser
from datetime import UTC, datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

# Configuration
REPO = "pymc-labs/CausalPy"
DEFAULT_PORT = 8765

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
}


def run_gh_command(args):
    """Run a gh command and return parsed JSON"""
    try:
        result = subprocess.run(
            ["gh"] + args, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout) if result.stdout else None
    except Exception:
        return None


def get_relative_time(updated_at_str):
    """Convert ISO timestamp to relative time string"""
    updated = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
    now = datetime.now(UTC)
    diff = now - updated

    if diff.days == 0:
        hours = diff.seconds // 3600
        if hours == 0:
            minutes = diff.seconds // 60
            return f"{minutes} min ago" if minutes != 1 else "1 min ago"
        return f"{hours}h ago" if hours != 1 else "1h ago"
    elif diff.days == 1:
        return "yesterday"
    elif diff.days < 7:
        return f"{diff.days}d ago"
    else:
        return f"{diff.days // 7}w ago"


def get_number_from_url(url):
    """Extract issue/PR number from API URL"""
    if not url:
        return None
    return url.rstrip("/").split("/")[-1]


def get_html_url(notification):
    """Generate the GitHub HTML URL for a notification"""
    url = notification.get("url")
    ntype = notification["type"]

    if not url:
        if ntype in ["WorkflowRun", "CheckSuite"]:
            return f"https://github.com/{REPO}/actions"
        return f"https://github.com/{REPO}"

    number = get_number_from_url(url)
    if ntype == "PullRequest":
        return f"https://github.com/{REPO}/pull/{number}"
    else:
        return f"https://github.com/{REPO}/issues/{number}"


def get_tags(notification, details):
    """Determine smart tags for a notification"""
    tags = []
    reason = notification["reason"]
    ntype = notification["type"]

    # Handle CI/Workflow notifications
    if ntype in ["WorkflowRun", "CheckSuite"]:
        title = notification.get("title", "").lower()
        if "failed" in title:
            tags.append(("🔴", "CI FAILED"))
        elif "cancelled" in title:
            tags.append(("⚪", "CI CANCELLED"))
        if reason == "approval_requested":
            tags.append(("🔴", "NEEDS RESPONSE"))
        else:
            tags.append(("🔧", "CI ACTIVITY"))
        return tags

    # Action-required tags
    if reason == "mention":
        tags.append(("🔴", "NEEDS RESPONSE"))
    if reason == "review_requested":
        tags.append(("🔴", "NEEDS RESPONSE"))

    # Author tag
    if reason == "author":
        tags.append(("👤", "YOUR PR" if ntype == "PullRequest" else "YOUR ISSUE"))

    if ntype == "PullRequest" and details:
        # Merged/Closed status
        if details.get("mergedAt"):
            tags.append(("🎉", "PR MERGED"))
            tags.append(("🏁", "RESOLVED"))
        elif details.get("state") == "CLOSED":
            tags.append(("⛔", "PR CLOSED"))
            tags.append(("🏁", "RESOLVED"))

        # Review status
        reviews = details.get("reviews", [])
        if reviews:
            if any(r.get("state") == "APPROVED" for r in reviews):
                tags.append(("✅", "PR APPROVED"))
            if any(r.get("state") == "CHANGES_REQUESTED" for r in reviews):
                tags.append(("🔶", "CHANGES REQUESTED"))

        # CI status
        checks = details.get("statusCheckRollup", [])
        if checks:
            conclusions = [c.get("conclusion") or c.get("state") for c in checks]
            conclusions = [
                c for c in conclusions if c and c not in ["SKIPPED", "NEUTRAL"]
            ]
            if conclusions:
                if any(c in ["FAILURE", "ERROR"] for c in conclusions):
                    tags.append(("🔴", "CI FAILED"))
                elif all(c in ["SUCCESS"] for c in conclusions):
                    tags.append(("🟢", "CI PASSING"))

        # Human feedback
        if reason == "author":
            reviews = details.get("reviews", [])
            human_reviews = [
                r
                for r in reviews
                if r.get("author", {}).get("login", "") not in BOT_USERNAMES
            ]
            if human_reviews and human_reviews[-1].get("state") == "COMMENTED":
                tags.append(("💬", "NEW REVIEW COMMENT"))

    elif ntype == "Issue" and details:
        if details.get("state") == "CLOSED":
            tags.append(("✅", "ISSUE CLOSED"))
            tags.append(("🏁", "RESOLVED"))

    # Comment tag
    if reason == "comment":
        tags.append(("💬", "NEW COMMENT"))

    # FYI only
    if reason == "subscribed" and not any(t[0] in ["🔴", "🔶"] for t in tags):
        tags.append(("👀", "FYI ONLY"))

    return tags


def categorize_notification(notification, tags):
    """Determine which category a notification belongs to"""
    reason = notification["reason"]
    ntype = notification["type"]
    tag_names = [t[1] for t in tags]
    is_resolved = "RESOLVED" in tag_names

    # Hidden
    if ntype in ["WorkflowRun", "CheckSuite"] and reason != "approval_requested":
        return "hidden"

    # Bot comments go to watching
    if "BOT COMMENT" in tag_names:
        return "watching"

    # Mentions always need response
    if reason == "mention":
        return "action"

    # Resolved items
    if is_resolved:
        return "resolved"

    # Review requests
    if reason == "review_requested":
        return "action"

    # Approval requests
    if reason == "approval_requested":
        return "action"

    # Changes requested
    if "CHANGES REQUESTED" in tag_names:
        return "action"

    # Author with feedback
    if reason == "author" and any(
        t in tag_names for t in ["NEW COMMENT", "NEW REVIEW COMMENT"]
    ):
        return "action"

    # By type
    if ntype == "PullRequest":
        return "pr"
    elif ntype == "Issue":
        return "issue"

    return "watching"


def load_notifications():
    """Load pre-enriched notifications from JSON.

    Expects notifications to be already enriched by fetch_notifications.py
    and summaries to be added by the Cursor agent.
    """
    # Try the enriched file first (new workflow)
    enriched_file = ".scratch/notifications-with-summaries.json"
    fallback_file = ".scratch/notifications-enriched.json"

    try:
        with open(enriched_file) as f:
            notifications = json.load(f)
            print(f"📂 Loaded from {enriched_file}")
    except FileNotFoundError:
        try:
            with open(fallback_file) as f:
                notifications = json.load(f)
                print(f"📂 Loaded from {fallback_file} (no summaries yet)")
        except FileNotFoundError:
            print("❌ No notification file found. Run fetch_notifications.py first.")
            return []

    # Update relative times (they may be stale)
    for n in notifications:
        if n.get("updated_at"):
            n["relative_time"] = get_relative_time(n["updated_at"])

    return notifications


def is_action_required(n):
    """Determine if a notification requires action."""
    reason = n.get("reason", "")
    tags = [t[1] for t in n.get("tags", [])]

    # Direct action triggers
    if reason in ["mention", "review_requested", "approval_requested"]:
        return True
    if "CHANGES REQUESTED" in tags:
        return True
    return reason == "author" and any(
        t in tags for t in ["NEW COMMENT", "NEW REVIEW COMMENT"]
    )


def generate_html(notifications):
    """Generate HTML page for notifications"""
    # Categorize by TYPE first, then we'll split by action within each type
    categories = {"pr": [], "issue": [], "watching": [], "resolved": [], "hidden": []}

    for n in notifications:
        cat = n["category"]
        # Map old "action" category back to pr/issue based on type
        if cat == "action":
            if n["type"] == "PullRequest":
                categories["pr"].append(n)
            elif n["type"] == "Issue":
                categories["issue"].append(n)
            else:
                categories["watching"].append(n)
        else:
            categories[cat].append(n)

    # Mark each notification as action_required
    for cat in categories.values():
        for n in cat:
            n["action_required"] = is_action_required(n)

    # Sort by updated_at
    for cat in categories.values():
        cat.sort(key=lambda x: x["updated_at"], reverse=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔔 GitHub Notifications: pymc-labs/CausalPy</title>
    <style>
        :root {{
            --bg: #f6f8fa;
            --card-bg: #ffffff;
            --border: #d0d7de;
            --text: #1f2328;
            --text-muted: #656d76;
            --accent: #0969da;
            --success: #1a7f37;
            --warning: #9a6700;
            --danger: #cf222e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding: 2rem;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}
        h1 {{ font-size: 1.5rem; display: flex; align-items: center; gap: 0.5rem; }}
        .stats {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
        }}
        .stat {{
            background: var(--card-bg);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            border: 1px solid var(--border);
            font-size: 0.9rem;
        }}
        .stat-value {{ font-weight: 600; }}
        .section {{ margin-bottom: 2rem; }}
        .section-title {{
            font-size: 1.1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .notification {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            transition: opacity 0.3s, transform 0.3s;
        }}
        .notification.done {{
            opacity: 0.4;
            transform: scale(0.98);
        }}
        .notification-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
        }}
        .notification-title {{
            font-weight: 600;
            color: var(--accent);
            text-decoration: none;
        }}
        .notification-title:hover {{ text-decoration: underline; }}
        .notification-meta {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }}
        .tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.5rem;
        }}
        .notification-summary {{
            font-size: 0.9rem;
            color: var(--text);
            margin-top: 0.75rem;
            padding: 0.75rem 1rem;
            background: var(--bg);
            border-radius: 6px;
            border-left: 3px solid var(--accent);
            line-height: 1.5;
        }}
        .notification-summary strong {{
            color: var(--accent);
            font-weight: 600;
        }}
        .notification-summary em {{
            font-style: italic;
        }}
        .notification-summary code {{
            background: var(--border);
            padding: 0.1rem 0.3rem;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        .notification-summary blockquote {{
            border-left: 2px solid var(--border);
            margin: 0.5rem 0;
            padding-left: 0.75rem;
            color: var(--text-muted);
        }}
        .notification-summary p {{
            margin: 0.5rem 0;
        }}
        .notification-summary p:first-child {{
            margin-top: 0;
        }}
        .notification-summary p:last-child {{
            margin-bottom: 0;
        }}
        .action-required {{
            background: #fff8e6;
            border-left-color: var(--warning);
        }}
        .action-required strong {{
            color: var(--warning);
        }}
        .tag {{
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 999px;
            background: rgba(88, 166, 255, 0.15);
            color: var(--accent);
        }}
        .tag.danger {{ background: rgba(248, 81, 73, 0.15); color: var(--danger); }}
        .tag.warning {{ background: rgba(210, 153, 34, 0.15); color: var(--warning); }}
        .tag.success {{ background: rgba(63, 185, 80, 0.15); color: var(--success); }}
        .actions {{
            display: flex;
            gap: 0.5rem;
            flex-shrink: 0;
        }}
        .btn {{
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            border: 1px solid var(--border);
            background: var(--card-bg);
            color: var(--text);
            cursor: pointer;
            font-size: 0.8rem;
            transition: background 0.2s;
        }}
        .btn:hover {{ background: var(--border); }}
        .btn.done {{ background: var(--success); color: #000; border-color: var(--success); }}
        .btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        .btn-primary {{
            background: var(--accent);
            color: white;
            border-color: var(--accent);
            text-decoration: none;
        }}
        .btn-primary:hover {{ opacity: 0.85; background: var(--accent); }}
        .refresh-btn {{
            padding: 0.5rem 1rem;
            background: var(--card-bg);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
        }}
        .refresh-btn:hover {{ background: var(--bg); }}
        .tabs {{
            display: flex;
            gap: 0.25rem;
            border-bottom: 2px solid var(--border);
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }}
        .tab {{
            padding: 0.75rem 1.25rem;
            cursor: pointer;
            border: none;
            background: transparent;
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 500;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .tab:hover {{ color: var(--text); }}
        .tab.active {{
            color: var(--accent);
            border-bottom-color: var(--accent);
        }}
        .tab-count {{
            background: var(--bg);
            padding: 0.15rem 0.5rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .tab.active .tab-count {{
            background: var(--accent);
            color: white;
        }}
        .tab.danger .tab-count {{
            background: var(--danger);
            color: white;
        }}
        .tab-panel {{
            display: none;
        }}
        .tab-panel.active {{
            display: block;
        }}
        .resolved-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        .resolved-table th, .resolved-table td {{
            text-align: left;
            padding: 0.5rem;
            border-bottom: 1px solid var(--border);
        }}
        .resolved-table a {{ color: var(--accent); text-decoration: none; }}
        .resolved-table a:hover {{ text-decoration: underline; }}
        .hidden-info {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-align: center;
            padding: 1rem;
        }}
        .empty-state {{
            text-align: center;
            padding: 3rem;
            color: var(--text-muted);
        }}
        .subheading {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-muted);
            margin: 1.5rem 0 0.75rem 0;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .subheading:first-child {{
            margin-top: 0;
        }}
        .subheading.action {{
            color: var(--danger);
            border-bottom-color: var(--danger);
        }}
        .subheading .count {{
            background: var(--bg);
            padding: 0.15rem 0.5rem;
            border-radius: 999px;
            font-size: 0.75rem;
        }}
        .subheading.action .count {{
            background: var(--danger);
            color: white;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔔 GitHub Notifications: pymc-labs/CausalPy</h1>
            <button class="refresh-btn" id="refresh-btn" onclick="refreshData()">🔄 Refresh</button>
        </header>

        <div class="tabs">
            <button class="tab active{" danger" if any(n.get("action_required") for n in categories["pr"]) else ""}" onclick="showTab('pr')">
                📋 PRs <span class="tab-count">{len(categories["pr"])}</span>
            </button>
            <button class="tab{" danger" if any(n.get("action_required") for n in categories["issue"]) else ""}" onclick="showTab('issue')">
                📝 Issues <span class="tab-count">{len(categories["issue"])}</span>
            </button>
            <button class="tab" onclick="showTab('watching')">
                👀 Watching <span class="tab-count">{len(categories["watching"])}</span>
            </button>
            <button class="tab" onclick="showTab('resolved')">
                🏁 Resolved <span class="tab-count">{len(categories["resolved"])}</span>
            </button>
        </div>
"""

    def render_notification(n, idx):
        import html as html_module

        tags_html = ""
        for emoji, name in n["tags"]:
            css_class = "tag"
            if "FAILED" in name or "NEEDS" in name:
                css_class += " danger"
            elif "CHANGES" in name:
                css_class += " warning"
            elif "PASSING" in name or "APPROVED" in name or "MERGED" in name:
                css_class += " success"
            tags_html += f'<span class="{css_class}">{emoji} {name}</span>'

        type_prefix = "PR" if n["type"] == "PullRequest" else "Issue"
        title = (
            f"{type_prefix} #{n['number']}: {n['title']}"
            if n.get("number")
            else n["title"]
        )

        # Get summary - store as data attribute for markdown rendering
        summary = n.get("summary", "")
        # Escape for HTML attribute (not content)
        summary_escaped = html_module.escape(summary).replace('"', "&quot;")
        action_class = " action-required" if n.get("category") == "action" else ""
        summary_html = (
            f'<div class="notification-summary{action_class}" data-markdown="{summary_escaped}"></div>'
            if summary
            else ""
        )

        return f'''
        <div class="notification" id="notif-{n["id"]}">
            <div class="notification-header">
                <div>
                    <a href="{n["html_url"]}" target="_blank" class="notification-title">{title}</a>
                    <div class="notification-meta">{n["relative_time"]} · {n.get("reason", "").replace("_", " ")}</div>
                    <div class="tags">{tags_html}</div>
                    {summary_html}
                </div>
                <div class="actions">
                    <button class="btn" onclick="markDone('{n["id"]}')">✓ Done</button>
                </div>
            </div>
        </div>
'''

    def render_tab_with_subgroups(tab_id, items, is_active=False):
        """Render a tab panel with Action Required and FYI subgroups."""
        active_class = " active" if is_active else ""
        html_out = f'<div id="tab-{tab_id}" class="tab-panel{active_class}">'

        if not items:
            html_out += '<div class="empty-state">No notifications</div>'
            html_out += "</div>"
            return html_out

        # Split into action required vs FYI
        action_items = [n for n in items if n.get("action_required")]
        fyi_items = [n for n in items if not n.get("action_required")]

        # Sort each by updated_at
        action_items.sort(key=lambda x: x["updated_at"], reverse=True)
        fyi_items.sort(key=lambda x: x["updated_at"], reverse=True)

        if action_items:
            html_out += f'<div class="subheading action">🔴 Action Required <span class="count">{len(action_items)}</span></div>'
            for i, n in enumerate(action_items):
                html_out += render_notification(n, i)

        if fyi_items:
            html_out += f'<div class="subheading">👀 FYI <span class="count">{len(fyi_items)}</span></div>'
            for i, n in enumerate(fyi_items):
                html_out += render_notification(n, i)

        html_out += "</div>"
        return html_out

    # Tab: PRs
    html += render_tab_with_subgroups("pr", categories["pr"], is_active=True)

    # Tab: Issues
    html += render_tab_with_subgroups("issue", categories["issue"])

    # Tab: Watching (all FYI by nature)
    html += '<div id="tab-watching" class="tab-panel">'
    if categories["watching"]:
        categories["watching"].sort(key=lambda x: x["updated_at"], reverse=True)
        for i, n in enumerate(categories["watching"]):
            html += render_notification(n, i)
    else:
        html += '<div class="empty-state">No watched notifications</div>'
    html += "</div>"

    # Tab: Resolved
    html += '<div id="tab-resolved" class="tab-panel">'
    if categories["resolved"]:
        html += """<table class="resolved-table">
            <tr><th>#</th><th>Title</th><th>Actions</th></tr>
"""
        for n in categories["resolved"]:
            number = n.get("number", "?")
            title_truncated = n["title"][:50] + ("..." if len(n["title"]) > 50 else "")
            html += f'''<tr id="notif-{n["id"]}">
                <td><a href="{n["html_url"]}" target="_blank">#{number}</a></td>
                <td>{title_truncated}</td>
                <td><button class="btn" onclick="markDone('{n["id"]}')">✓ Done</button></td>
            </tr>'''
        html += "</table>"
    else:
        html += '<div class="empty-state">No resolved items</div>'
    html += "</div>"

    # Hidden info
    if categories["hidden"]:
        html += f'<div class="hidden-info">🙈 {len(categories["hidden"])} notification(s) hidden (CI activity)</div>'

    html += """
        <script>
            function showTab(tabName) {
                // Hide all panels
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
                // Deactivate all tabs
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));

                // Show selected panel
                document.getElementById('tab-' + tabName).classList.add('active');
                // Activate selected tab
                event.target.closest('.tab').classList.add('active');
            }

            // Render markdown summaries on page load
            document.addEventListener('DOMContentLoaded', function() {
                document.querySelectorAll('[data-markdown]').forEach(el => {
                    const md = el.getAttribute('data-markdown');
                    if (md && typeof marked !== 'undefined') {
                        el.innerHTML = marked.parse(md);
                    } else if (md) {
                        el.textContent = md;
                    }
                });
            });

            function refreshData() {
                const btn = document.getElementById('refresh-btn');
                const originalText = btn.textContent;
                btn.disabled = true;
                btn.style.minWidth = '220px';

                // Remember current tab before refresh
                const currentTab = document.querySelector('.tab-btn.active')?.dataset.tab || 'pr';
                sessionStorage.setItem('activeTab', currentTab);

                // Use Server-Sent Events for progress updates
                const evtSource = new EventSource('/api/refresh-stream');

                evtSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    btn.textContent = data.message;

                    if (data.status === 'done') {
                        evtSource.close();
                        setTimeout(() => location.reload(), 500);
                    } else if (data.status === 'error') {
                        evtSource.close();
                        setTimeout(() => {
                            btn.textContent = originalText;
                            btn.disabled = false;
                            btn.style.minWidth = '';
                        }, 2000);
                    }
                };

                evtSource.onerror = function() {
                    evtSource.close();
                    btn.textContent = '❌ Connection error';
                    setTimeout(() => {
                        btn.textContent = originalText;
                        btn.disabled = false;
                        btn.style.minWidth = '';
                    }, 2000);
                };
            }

            // Restore tab on page load
            document.addEventListener('DOMContentLoaded', function() {
                const savedTab = sessionStorage.getItem('activeTab');
                if (savedTab) {
                    showTab(savedTab);
                }
            });

            async function markDone(threadId) {
                const btn = event.target;
                btn.disabled = true;
                btn.textContent = '...';

                try {
                    const res = await fetch('/api/done/' + threadId, { method: 'POST' });
                    if (res.ok) {
                        const el = document.getElementById('notif-' + threadId);
                        el.classList.add('done');
                        btn.textContent = '✓ Done!';
                        btn.classList.add('done');
                    } else {
                        btn.textContent = '✗ Failed';
                    }
                } catch (e) {
                    btn.textContent = '✗ Error';
                }
            }
        </script>
    </div>
</body>
</html>
"""
    return html


class NotificationHandler(SimpleHTTPRequestHandler):
    """HTTP handler for notification server"""

    notifications = []

    def do_GET(self):
        if self.path == "/" or self.path == "/notifications":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = generate_html(self.notifications)
            self.wfile.write(html.encode())
        elif self.path == "/api/refresh-stream":
            self.handle_refresh_stream()
        else:
            self.send_error(404)

    def send_sse(self, status, message):
        """Send a Server-Sent Event."""
        import json

        data = json.dumps({"status": status, "message": message})
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def handle_refresh_stream(self):
        """Handle refresh with streaming progress updates."""
        import os

        self.send_response(200)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Get the project root (parent of .cursor/commands)
        project_root = Path(__file__).parent.parent.parent

        try:
            # Step 1: Fetching from GitHub
            self.send_sse("progress", "⏳ Fetching from GitHub...")

            result = subprocess.run(
                [
                    "python",
                    str(project_root / ".cursor/commands/fetch_notifications.py"),
                    "--days",
                    "7",
                ],
                capture_output=True,
                text=True,
                cwd=str(project_root),
                env=os.environ.copy(),  # Pass full environment for gh auth
            )
            if result.returncode != 0:
                print(f"Fetch error: {result.stderr}")
                self.send_sse("error", "❌ Failed to fetch")
                return

            # Step 2: Enriching notifications
            self.send_sse("progress", "📋 Enriching notifications...")

            # Count notifications
            try:
                with open(project_root / ".scratch/notifications-enriched.json") as f:
                    import json

                    notifications = json.load(f)
                    count = len(notifications)
            except Exception:
                count = "?"

            # Step 3: Generating summaries
            self.send_sse("progress", f"🤖 Generating summaries ({count} items)...")

            result = subprocess.run(
                [
                    "python",
                    str(project_root / ".cursor/commands/generate_summaries.py"),
                ],
                capture_output=True,
                text=True,
                cwd=str(project_root),
                env=os.environ.copy(),
            )
            if result.returncode != 0:
                print(f"Summary error: {result.stderr}")
                self.send_sse("error", "❌ Failed to generate summaries")
                return

            # Step 4: Reloading
            self.send_sse("progress", "🔄 Reloading data...")

            # Reload notifications
            NotificationHandler.notifications = load_notifications()

            # Done!
            self.send_sse("done", "✅ Done! Reloading...")

        except Exception as e:
            print(f"Error during refresh: {e}")
            self.send_sse("error", f"❌ Error: {str(e)[:30]}")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/done/"):
            thread_id = parsed.path.split("/")[-1]
            success = self.mark_notification_done(thread_id)
            if success:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            else:
                self.send_error(500, "Failed to mark as done")
        elif parsed.path == "/api/refresh":
            success = self.refresh_notifications()
            if success:
                # Reload the notifications
                NotificationHandler.notifications = load_notifications()
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                count = len(NotificationHandler.notifications)
                self.wfile.write(f'{{"status": "ok", "count": {count}}}'.encode())
            else:
                self.send_error(500, "Failed to refresh notifications")
        else:
            self.send_error(404)

    def refresh_notifications(self):
        """Fetch fresh notifications from GitHub and regenerate summaries."""
        print("🔄 Refreshing notifications from GitHub...")

        # Run fetch script
        result = subprocess.run(
            ["python", ".cursor/commands/fetch_notifications.py", "--days", "7"],
            capture_output=True,
            text=True,
            cwd=Path(".").resolve(),
        )
        if result.returncode != 0:
            print(f"❌ Fetch failed: {result.stderr}")
            return False

        print("📝 Generating summaries...")

        # Run summary generation
        result = subprocess.run(
            ["python", "generate_summaries.py"],
            capture_output=True,
            text=True,
            cwd=Path(".scratch").resolve(),
        )
        if result.returncode != 0:
            print(f"❌ Summary generation failed: {result.stderr}")
            return False

        print("✅ Refresh complete!")
        return True

    def mark_notification_done(self, thread_id):
        """Mark a notification as Done (archived) via GitHub API.

        Uses: DELETE /notifications/threads/{thread_id}
        Docs: https://docs.github.com/en/rest/activity/notifications#mark-a-thread-as-done

        Discussion about this API feature:
        https://github.com/orgs/community/discussions/50224
        """
        print(f"📤 Marking notification {thread_id} as done...")

        # DELETE marks the thread as "Done" (archived)
        result = subprocess.run(
            ["gh", "api", "-X", "DELETE", f"/notifications/threads/{thread_id}"],
            capture_output=True,
        )
        print(f"   DELETE result: returncode={result.returncode}")

        # DELETE returns 204 No Content on success
        success = result.returncode == 0
        print(f"   Success: {success}")
        return success

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def main():
    parser = argparse.ArgumentParser(description="Notification server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    print("🔄 Loading notifications...", file=sys.stderr)
    NotificationHandler.notifications = load_notifications()
    print(
        f"📊 Loaded {len(NotificationHandler.notifications)} notifications",
        file=sys.stderr,
    )

    server = HTTPServer(("localhost", args.port), NotificationHandler)
    url = f"http://localhost:{args.port}/"

    print(f"🚀 Server running at {url}", file=sys.stderr)
    print("   Press Ctrl+C to stop", file=sys.stderr)

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
