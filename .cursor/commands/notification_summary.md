# GitHub Notifications Summary

Generate a morning briefing of GitHub notifications for the current repository with AI-generated summaries.

## Purpose

This command fetches your GitHub notifications, analyzes them for actionability, generates intelligent AI summaries, and presents an interactive web UI. Think of it as your "inbox zero" tool for GitHub.

## Configuration

The command **auto-detects** the repository from your git remote. No configuration needed for most users!

### Bot Blacklist (YAML)

Edit `.cursor/commands/notification_bots.yml` to customize which bots to **ignore**:

```yaml
# Bot Blacklist - notifications from these usernames are IGNORED
# (This is a blacklist, not a whitelist)

# Code coverage bots
- codecov
- codecov[bot]

# CI/CD bots
- pre-commit-ci
- github-actions[bot]

# Add your custom bots:
- my-custom-bot
```

### General Settings (JSON)

Create `.cursor/notification_config.json` for other settings:

```json
{
  "repo": "org/repo",
  "default_days": 7,
  "server_port": 8765
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `repo` | Auto-detected from git remote | Override repository (e.g., "pymc-labs/CausalPy") |
| `default_days` | 7 | Days to look back for notifications |
| `server_port` | 8765 | Local server port |

### Config Files Summary

| File | Purpose |
|------|---------|
| `.cursor/commands/notification_bots.yml` | **Bot blacklist** - usernames to IGNORE |
| `.cursor/notification_config.json` | General settings (repo, days, port) |

## Parameters

When invoking this command, the user may optionally specify:
- **Time window**: How far back to look (e.g., "last 24 hours", "last 7 days")
  - Default: Last 7 days if not specified

## Prerequisites

Before starting, verify the GitHub CLI is available:

```bash
gh --version
gh auth status
```

If not authenticated, guide: `gh auth login`

**Important:** All `gh` commands require `required_permissions: ["all"]`

## Workflow

### Phase 1: Parse User Request

Determine the time window from the user's request:
- "last 24 hours" or "today" → 1 day
- "last 3 days" → 3 days
- "last week" or "last 7 days" → 7 days (default)
- "last 2 weeks" → 14 days
- "last month" → 30 days

### Phase 2: Fetch and Enrich Notifications

Run the fetch script to get notifications and enrich them with PR/Issue details:

```bash
python .cursor/commands/fetch_notifications.py --days DAYS
```

This creates `.scratch/notifications-enriched.json` with:
- Basic notification info (id, reason, type, title)
- PR/Issue details (state, reviews, comments, CI status)
- Smart tags (NEEDS RESPONSE, PR MERGED, CI FAILED, etc.)
- Category assignments (action, pr, issue, watching, resolved)

### Phase 3: Generate AI Summaries ⭐

This is where you add value! Read the enriched JSON and generate a summary for each notification.

1. **Read the file:**
   ```bash
   cat .scratch/notifications-enriched.json
   ```

2. **For each notification, generate a summary** (max 500 words) that answers:
   - What is this about?
   - What action (if any) is needed from me?
   - What's the current status?
   - Key discussion points or decisions made

3. **Consider the context:**
   - For `reason: "mention"` → Focus on what was asked/discussed
   - For `reason: "review_requested"` → Summarize the PR changes and what to review
   - For `reason: "author"` → Summarize feedback received on your PR/issue
   - For `category: "resolved"` → Brief status (merged, closed, etc.)

4. **Use the details field** which contains:
   - `body`: The PR/issue description
   - `comments`: Array of comments with `author.login` and `body`
   - `reviews`: Array of reviews with `author.login`, `state`, and `body`

5. **Save the enriched data** with summaries added:

After generating all summaries, create `.scratch/notifications-with-summaries.json` with the same structure but with the `summary` field populated for each notification.

**Example format for saving:**
```json
[
  {
    "id": "12345",
    "title": "Add new feature",
    "type": "PullRequest",
    "reason": "review_requested",
    "category": "action",
    "tags": [["🔴", "NEEDS RESPONSE"]],
    "html_url": "https://github.com/org/repo/pull/123",
    "summary": "Your AI-generated summary here (max 500 words)...",
    ...
  }
]
```

### Phase 4: Launch Interactive Web UI

```bash
python .cursor/commands/notification_server.py
```

This will:
1. Load notifications from `.scratch/notifications-with-summaries.json`
2. Start a local server (default: http://localhost:8765)
3. Auto-open your browser
4. Show notifications in tabs with your AI summaries
5. Provide "Done" buttons that mark notifications as done on GitHub

### Phase 5: Present to User

After launching the web UI, offer follow-up actions:

> "I've analyzed your notifications and launched the interactive dashboard.
>
> **Quick Summary:**
> - 🔥 X items need your action
> - 📋 Y pull requests
> - 📝 Z issues
> - 🏁 N resolved items
>
> Would you like me to:
> 1. **Deep dive** into any specific notification?
> 2. **Draft a response** to a comment or review?
> 3. **Help review** a specific PR?"

## Smart Tags

| Tag | Emoji | Criteria |
|-----|-------|----------|
| `NEEDS RESPONSE` | 🔴 | Mentioned or review requested |
| `YOUR PR` / `YOUR ISSUE` | 👤 | You are the author |
| `PR APPROVED` | ✅ | Has approving reviews |
| `PR MERGED` | 🎉 | PR was merged |
| `CHANGES REQUESTED` | 🔶 | Reviews request changes |
| `CI FAILED` | 🔴 | Status checks failed |
| `CI PASSING` | 🟢 | Status checks passed |
| `NEW REVIEW COMMENT` | 💬 | Human feedback on your PR |
| `RESOLVED` | 🏁 | Closed or merged |

## Categories (Priority Order)

1. **🔥 Requires Your Action** - Mentions, review requests, changes requested
2. **📋 Pull Requests** - PR activity (not requiring immediate action)
3. **📝 Issues** - Issue activity
4. **👀 Watching** - Subscribed/FYI items
5. **🏁 Resolved** - Closed/merged items

## Files Used

| File | Purpose |
|------|---------|
| `.cursor/commands/notification_config.py` | Configuration auto-detection |
| `.cursor/commands/fetch_notifications.py` | Fetches and enriches notifications |
| `.cursor/commands/generate_summaries.py` | Template summary generator |
| `.cursor/commands/notification_server.py` | Serves the interactive web UI |
| `.scratch/notifications-enriched.json` | Enriched notifications (no summaries) |
| `.scratch/notifications-with-summaries.json` | With AI summaries added |
| `.cursor/notification_config.json` | (Optional) Custom configuration |

## Example Summary Generation

For a notification like:
```json
{
  "title": "Add hierarchical DiD model",
  "reason": "mention",
  "details": {
    "body": "This PR adds support for hierarchical difference-in-differences...",
    "comments": [
      {"author": {"login": "reviewer"}, "body": "@you what do you think about the prior choices?"}
    ]
  }
}
```

Generate a summary like:
> "A reviewer is asking for your input on prior choices in a new hierarchical DiD model PR. The PR adds support for panel data with multiple treated units. Specifically, they want your opinion on the priors for the random effects variance. **Action needed:** Review and respond to the question about prior specifications."

## Error Handling

- **No notifications:** Report "No notifications found in the specified time window."
- **Repository not detected:** Create `.cursor/notification_config.json` with `{"repo": "org/repo"}`
- **Script not found:** Ensure `.cursor/commands/fetch_notifications.py` exists
- **Empty enriched file:** Run fetch_notifications.py first
