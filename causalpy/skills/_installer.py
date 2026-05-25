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
"""Skill installer: reads bundled user skills and writes them to platform-specific locations."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Literal

from causalpy.skills._platforms import claude, copilot, cursor, generic, windsurf
from causalpy.version import __version__

Platform = Literal["cursor", "claude", "copilot", "windsurf", "generic"]

_PLATFORM_ADAPTERS: dict[Platform, type] = {
    "cursor": cursor.CursorAdapter,
    "claude": claude.ClaudeAdapter,
    "copilot": copilot.CopilotAdapter,
    "windsurf": windsurf.WindsurfAdapter,
    "generic": generic.GenericAdapter,
}

_SKILL_DIRS = [
    "designing-experiments",
    "performing-causal-analysis",
    "running-placebo-analysis",
]

VERSION_STAMP = f"<!-- causalpy-skills v{__version__} -->"


def _skills_package() -> importlib.resources.abc.Traversable:
    """Return the traversable root of the bundled skills package data."""
    return importlib.resources.files("causalpy.skills")


def list_skills() -> list[str]:
    """Return names of all bundled user skills."""
    return list(_SKILL_DIRS)


def _read_skill_tree(skill_name: str) -> dict[str, str]:
    """Read all Markdown files for a single skill, returning {relative_path: content}."""
    root = _skills_package() / skill_name
    files: dict[str, str] = {}

    def _walk(node: importlib.resources.abc.Traversable, prefix: str) -> None:
        """Recursively collect ``.md`` files from *node* into *files*."""
        for child in node.iterdir():
            rel = f"{prefix}/{child.name}" if prefix else child.name
            if child.is_file() and child.name.endswith(".md"):
                files[rel] = child.read_text(encoding="utf-8")
            elif child.is_dir():
                _walk(child, rel)

    _walk(root, "")
    return files


def detect_platforms(project_dir: Path) -> list[Platform]:
    """Auto-detect which AI platforms are configured in *project_dir*."""
    detected: list[Platform] = []
    if (project_dir / ".cursor").is_dir():
        detected.append("cursor")
    if (project_dir / ".claude").is_dir():
        detected.append("claude")
    if (project_dir / ".github" / "copilot-instructions.md").exists():
        detected.append("copilot")
    if (project_dir / ".windsurf").is_dir() or (
        project_dir / ".windsurfrules"
    ).exists():
        detected.append("windsurf")
    return detected


def install(
    project_dir: str | Path = ".",
    platform: Platform | None = None,
) -> dict[str, list[Path]]:
    """Install user skills into *project_dir* for the given *platform*.

    If *platform* is ``None``, auto-detect from the project directory.
    Falls back to ``"generic"`` if no platform is detected.

    Returns a mapping of ``{platform: [written_paths]}``.
    """
    project = Path(project_dir).resolve()
    if not project.is_dir():
        msg = f"Project directory does not exist: {project}"
        raise FileNotFoundError(msg)

    platforms: list[Platform]
    if platform is not None:
        platforms = [platform]
    else:
        platforms = detect_platforms(project) or ["generic"]

    skills_data: dict[str, dict[str, str]] = {}
    for name in _SKILL_DIRS:
        skills_data[name] = _read_skill_tree(name)

    result: dict[str, list[Path]] = {}
    for plat in platforms:
        adapter = _PLATFORM_ADAPTERS[plat](project, VERSION_STAMP)
        result[plat] = adapter.install(skills_data)

    return result


def uninstall(
    project_dir: str | Path = ".",
    platform: Platform | None = None,
) -> dict[str, list[Path]]:
    """Remove previously installed skills from *project_dir*.

    Returns a mapping of ``{platform: [removed_paths]}``.
    """
    project = Path(project_dir).resolve()
    if not project.is_dir():
        msg = f"Project directory does not exist: {project}"
        raise FileNotFoundError(msg)

    platforms: list[Platform]
    if platform is not None:
        platforms = [platform]
    else:
        platforms = detect_platforms(project) or ["generic"]

    result: dict[str, list[Path]] = {}
    for plat in platforms:
        adapter = _PLATFORM_ADAPTERS[plat](project, VERSION_STAMP)
        result[plat] = adapter.uninstall()

    return result


def check_version(
    project_dir: str | Path = ".",
) -> dict[str, str | None]:
    """Return a per-platform mapping of installed skill versions.

    Each key is a platform name; the value is the version string found
    for that platform, or ``None`` if no skills are installed there.
    Only platforms that have skills installed are included.
    """
    project = Path(project_dir).resolve()
    result: dict[str, str | None] = {}
    for plat_name, adapter_cls in _PLATFORM_ADAPTERS.items():
        adapter = adapter_cls(project, VERSION_STAMP)
        version = adapter.get_installed_version()
        if version is not None:
            result[plat_name] = version
    return result
