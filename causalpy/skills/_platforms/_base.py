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
"""Base class for platform adapters."""

from __future__ import annotations

import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class BasePlatformAdapter(ABC):
    """Common interface that every platform adapter implements."""

    def __init__(self, project_dir: Path, version_stamp: str) -> None:
        """Initialise with *project_dir* and the HTML-comment *version_stamp*."""
        self.project_dir = project_dir
        self.version_stamp = version_stamp

    @abstractmethod
    def install(self, skills_data: dict[str, dict[str, str]]) -> list[Path]:
        """Write skills to the platform-specific location. Return written paths."""

    @abstractmethod
    def uninstall(self) -> list[Path]:
        """Remove previously installed skills. Return removed paths."""

    @abstractmethod
    def get_installed_version(self) -> str | None:
        """Return the version string from an existing install, or ``None``."""

    @staticmethod
    def _extract_version(text: str) -> str | None:
        """Extract the CausalPy version from a version-stamp comment in *text*."""
        m = re.search(r"<!-- causalpy-skills v([\d.]+\S*) -->", text)
        return m.group(1) if m else None

    def _stamp(self, content: str) -> str:
        """Prepend version stamp to content."""
        return f"{self.version_stamp}\n{content}"

    def _rmtree_if_exists(self, path: Path) -> bool:
        """Remove *path* (file or directory) if it exists. Return whether it existed."""
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True
        return False
