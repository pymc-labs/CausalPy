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
"""GitHub Copilot platform adapter — appends a marked section to ``.github/copilot-instructions.md``."""

from __future__ import annotations

import re
from pathlib import Path

from causalpy.skills._platforms._base import BasePlatformAdapter

_BEGIN = "<!-- BEGIN causalpy-skills -->"
_END = "<!-- END causalpy-skills -->"


class CopilotAdapter(BasePlatformAdapter):
    """Install/uninstall CausalPy skills for GitHub Copilot."""

    @property
    def _instructions_file(self) -> Path:
        """Return the path to ``copilot-instructions.md``."""
        return self.project_dir / ".github" / "copilot-instructions.md"

    def _build_section(self, skills_data: dict[str, dict[str, str]]) -> str:
        """Build the fenced Markdown section to insert into the instructions file."""
        parts = [_BEGIN, self.version_stamp, ""]
        parts.append("# CausalPy Agent Skills\n")
        for _skill_name, files in skills_data.items():
            if "SKILL.md" in files:
                parts.append(files["SKILL.md"])
                parts.append("")
            for rel_path, content in sorted(files.items()):
                if rel_path != "SKILL.md":
                    parts.append(content)
                    parts.append("")
        parts.append(_END)
        return "\n".join(parts)

    def install(self, skills_data: dict[str, dict[str, str]]) -> list[Path]:
        """Append (or replace) the CausalPy section in ``copilot-instructions.md``."""
        section = self._build_section(skills_data)
        self._instructions_file.parent.mkdir(parents=True, exist_ok=True)

        if self._instructions_file.exists():
            existing = self._instructions_file.read_text(encoding="utf-8")
            pattern = re.compile(
                rf"{re.escape(_BEGIN)}.*?{re.escape(_END)}",
                re.DOTALL,
            )
            if pattern.search(existing):
                updated = pattern.sub(section, existing)
            else:
                updated = existing.rstrip() + "\n\n" + section + "\n"
        else:
            updated = section + "\n"

        self._instructions_file.write_text(updated, encoding="utf-8")
        return [self._instructions_file]

    def uninstall(self) -> list[Path]:
        """Remove the CausalPy section from ``copilot-instructions.md``."""
        if not self._instructions_file.exists():
            return []
        existing = self._instructions_file.read_text(encoding="utf-8")
        pattern = re.compile(
            rf"\n*{re.escape(_BEGIN)}.*?{re.escape(_END)}\n*",
            re.DOTALL,
        )
        updated = pattern.sub("", existing)
        if updated.strip():
            self._instructions_file.write_text(updated, encoding="utf-8")
        else:
            self._instructions_file.unlink()
        return [self._instructions_file]

    def get_installed_version(self) -> str | None:
        """Read the version stamp from the instructions file, if present."""
        if not self._instructions_file.exists():
            return None
        content = self._instructions_file.read_text(encoding="utf-8")
        return self._extract_version(content)
