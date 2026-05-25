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
"""Generic platform adapter — writes a single ``llms-causalpy.txt`` context document."""

from __future__ import annotations

from pathlib import Path

from causalpy.skills._platforms._base import BasePlatformAdapter

_FILENAME = "llms-causalpy.txt"


def build_llms_txt(
    skills_data: dict[str, dict[str, str]],
    version_stamp: str,
) -> str:
    """Build the concatenated llms.txt content from all skills.

    This is also used by the docs-build ``llms.txt`` generator, so it is
    a module-level function rather than only an adapter method.
    """
    parts = [
        version_stamp,
        "",
        "# CausalPy — AI Agent Context",
        "",
        "> Causal inference for quasi-experiments in Python. "
        "This document teaches an AI agent how to use CausalPy to perform "
        "causal analyses (Difference-in-Differences, Interrupted Time Series, "
        "Synthetic Control, Regression Discontinuity, and more).",
        "",
        "---",
        "",
    ]
    for _skill_name, files in skills_data.items():
        if "SKILL.md" in files:
            parts.append(files["SKILL.md"])
            parts.append("")
        for rel_path, content in sorted(files.items()):
            if rel_path != "SKILL.md":
                parts.append(content)
                parts.append("")
        parts.append("---")
        parts.append("")

    return "\n".join(parts)


class GenericAdapter(BasePlatformAdapter):
    """Install/uninstall a generic ``llms-causalpy.txt`` context file."""

    @property
    def _output_file(self) -> Path:
        """Return the path to the output ``llms-causalpy.txt`` file."""
        return self.project_dir / _FILENAME

    def install(self, skills_data: dict[str, dict[str, str]]) -> list[Path]:
        """Write all skills as a single ``llms-causalpy.txt`` file."""
        content = build_llms_txt(skills_data, self.version_stamp)
        self._output_file.write_text(content, encoding="utf-8")
        return [self._output_file]

    def uninstall(self) -> list[Path]:
        """Remove the ``llms-causalpy.txt`` file if it exists."""
        if self._output_file.exists():
            self._output_file.unlink()
            return [self._output_file]
        return []

    def get_installed_version(self) -> str | None:
        """Read the version stamp from the ``llms-causalpy.txt`` file."""
        if not self._output_file.exists():
            return None
        content = self._output_file.read_text(encoding="utf-8")
        return self._extract_version(content)
