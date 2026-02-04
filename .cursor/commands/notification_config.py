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
"""
Configuration for the GitHub notification summary command.

Provides auto-detection of org/repo from git remote, with optional
config file overrides for customization.

Configuration files:
- .cursor/commands/notification_bots.yml - Bot blacklist (usernames to IGNORE)
- .cursor/notification_config.json - General settings (optional)
"""

import json
import re
import subprocess
from pathlib import Path

# Default settings
DEFAULT_DAYS = 7
DEFAULT_SERVER_PORT = 8765

# Fallback bot blacklist (used if YAML file not found)
FALLBACK_BOT_BLACKLIST = {
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


def get_project_root() -> Path:
    """Get the project root directory (where .git is located)."""
    # Start from this file's location and go up
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    # Fallback to cwd
    return Path.cwd()


def get_commands_dir() -> Path:
    """Get the .cursor/commands directory."""
    return Path(__file__).resolve().parent


def load_bot_blacklist() -> set[str]:
    """
    Load the bot blacklist from notification_bots.yml.

    This is a BLACKLIST - notifications from these usernames are IGNORED.
    Helps filter out noise from automated bot comments (codecov, dependabot, etc.)

    Returns:
        Set of usernames to ignore (case-sensitive as GitHub usernames are).
    """
    commands_dir = get_commands_dir()
    bots_file = commands_dir / "notification_bots.yml"

    if bots_file.exists():
        try:
            # Simple YAML list parsing (avoids PyYAML dependency)
            bots = set()
            with open(bots_file) as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue
                    # Parse list items (- username)
                    if line.startswith("- "):
                        username = line[2:].strip()
                        # Remove inline comments
                        if "#" in username:
                            username = username.split("#")[0].strip()
                        if username:
                            bots.add(username)
            if bots:
                return bots
        except Exception as e:
            print(f"Warning: Failed to load bot blacklist from {bots_file}: {e}")

    # Fallback to hardcoded defaults
    return FALLBACK_BOT_BLACKLIST.copy()


def parse_git_remote(remote_url: str) -> tuple[str, str] | None:
    """
    Parse org and repo from a git remote URL.

    Supports:
    - https://github.com/org/repo.git
    - https://github.com/org/repo
    - git@github.com:org/repo.git
    - git@github.com:org/repo
    """
    # HTTPS format
    https_match = re.match(
        r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", remote_url
    )
    if https_match:
        return https_match.group(1), https_match.group(2)

    # SSH format
    ssh_match = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", remote_url)
    if ssh_match:
        return ssh_match.group(1), ssh_match.group(2)

    return None


def detect_repo_from_git() -> str | None:
    """Auto-detect org/repo from git remote origin."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            remote_url = result.stdout.strip()
            parsed = parse_git_remote(remote_url)
            if parsed:
                org, repo = parsed
                return f"{org}/{repo}"
    except Exception:
        pass
    return None


def load_config() -> dict:
    """
    Load configuration with the following priority:
    1. .cursor/notification_config.json (if exists) - for repo, days, port
    2. .cursor/commands/notification_bots.yml - for bot blacklist
    3. Auto-detect repo from git remote
    4. Built-in defaults

    Returns a dict with:
    - repo: str (e.g., "pymc-labs/CausalPy")
    - bot_blacklist: set[str] - usernames to IGNORE
    - default_days: int
    - server_port: int
    """
    project_root = get_project_root()
    config_file = project_root / ".cursor" / "notification_config.json"

    # Start with defaults
    config = {
        "repo": None,
        "bot_blacklist": load_bot_blacklist(),
        "default_days": DEFAULT_DAYS,
        "server_port": DEFAULT_SERVER_PORT,
    }

    # Try to load JSON config file for general settings
    if config_file.exists():
        try:
            with open(config_file) as f:
                user_config = json.load(f)

            # Override with user settings
            if "repo" in user_config:
                config["repo"] = user_config["repo"]
            if "default_days" in user_config:
                config["default_days"] = user_config["default_days"]
            if "server_port" in user_config:
                config["server_port"] = user_config["server_port"]
            # Legacy support: bot_usernames in JSON extends the blacklist
            if "bot_usernames" in user_config:
                config["bot_blacklist"].update(user_config["bot_usernames"])
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")

    # Auto-detect repo if not specified
    if not config["repo"]:
        detected = detect_repo_from_git()
        if detected:
            config["repo"] = detected

    return config


def get_repo() -> str:
    """Get the repository in org/repo format."""
    config = load_config()
    repo = config.get("repo")
    if not repo:
        raise ValueError(
            "Could not determine repository. Either:\n"
            "1. Run from a git repository with a GitHub remote, or\n"
            '2. Create .cursor/notification_config.json with: {"repo": "org/repo"}'
        )
    return repo


def get_bot_blacklist() -> set[str]:
    """
    Get the set of bot usernames to IGNORE (blacklist).

    Notifications from these usernames are filtered out to reduce noise.
    Edit .cursor/commands/notification_bots.yml to customize.
    """
    return load_config()["bot_blacklist"]


# Alias for backwards compatibility
def get_bot_usernames() -> set[str]:
    """Alias for get_bot_blacklist() - returns usernames to ignore."""
    return get_bot_blacklist()


def get_default_days() -> int:
    """Get the default number of days to look back."""
    return load_config()["default_days"]


def get_server_port() -> int:
    """Get the server port."""
    return load_config()["server_port"]


if __name__ == "__main__":
    # Test the config
    print("=" * 60)
    print("GitHub Notification Summary - Configuration")
    print("=" * 60)
    config = load_config()
    print(f"\n📦 Repository: {config['repo'] or '(not detected)'}")
    print(f"📅 Default days: {config['default_days']}")
    print(f"🌐 Server port: {config['server_port']}")
    print(f"\n🤖 Bot Blacklist ({len(config['bot_blacklist'])} usernames to IGNORE):")
    for bot in sorted(config["bot_blacklist"]):
        print(f"   - {bot}")
    print("\n📁 Config files:")
    print(f"   - Bot blacklist: {get_commands_dir() / 'notification_bots.yml'}")
    print(
        f"   - General config: {get_project_root() / '.cursor/notification_config.json'}"
    )
