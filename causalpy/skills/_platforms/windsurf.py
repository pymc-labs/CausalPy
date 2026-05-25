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
"""Windsurf platform adapter — writes skills to ``.windsurf/skills/causalpy-<name>/``."""

from __future__ import annotations

from pathlib import Path

from causalpy.skills._platforms._base import BasePlatformAdapter

_PREFIX = "causalpy-"


class WindsurfAdapter(BasePlatformAdapter):
    """Install/uninstall CausalPy skills for Windsurf."""

    @property
    def _skills_root(self) -> Path:
        """Return the Windsurf skills directory path."""
        return self.project_dir / ".windsurf" / "skills"

    def install(self, skills_data: dict[str, dict[str, str]]) -> list[Path]:
        """Write each skill to ``.windsurf/skills/causalpy-<name>/``."""
        written: list[Path] = []
        self._skills_root.mkdir(parents=True, exist_ok=True)
        for skill_name, files in skills_data.items():
            skill_dir = self._skills_root / f"{_PREFIX}{skill_name}"
            skill_dir.mkdir(parents=True, exist_ok=True)
            for rel_path, content in files.items():
                out = skill_dir / rel_path
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(self._stamp(content), encoding="utf-8")
                written.append(out)
        return written

    def uninstall(self) -> list[Path]:
        """Remove all ``causalpy-*`` skill directories."""
        removed: list[Path] = []
        if not self._skills_root.exists():
            return removed
        for child in self._skills_root.iterdir():
            if child.is_dir() and child.name.startswith(_PREFIX):
                removed.append(child)
                self._rmtree_if_exists(child)
        return removed

    def get_installed_version(self) -> str | None:
        """Read the version stamp from an installed skill, if any."""
        if not self._skills_root.exists():
            return None
        for child in self._skills_root.iterdir():
            if child.is_dir() and child.name.startswith(_PREFIX):
                skill_md = child / "SKILL.md"
                if skill_md.exists():
                    return self._extract_version(skill_md.read_text(encoding="utf-8"))
        return None
