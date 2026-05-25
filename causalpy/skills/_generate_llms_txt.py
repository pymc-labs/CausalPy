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
"""Generate ``llms.txt`` from bundled user skills.

Run as a script to write ``llms.txt`` to a given output path::

    python -m causalpy.skills._generate_llms_txt docs/source/llms.txt

Or invoke ``make llms-txt`` from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

from causalpy.skills._installer import _SKILL_DIRS, VERSION_STAMP, _read_skill_tree
from causalpy.skills._platforms.generic import build_llms_txt


def generate(output: Path) -> None:
    """Build llms.txt from bundled skills and write to *output*."""
    skills_data: dict[str, dict[str, str]] = {}
    for name in _SKILL_DIRS:
        skills_data[name] = _read_skill_tree(name)

    content = build_llms_txt(skills_data, VERSION_STAMP)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"  Generated {output} ({len(content)} bytes)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m causalpy.skills._generate_llms_txt <output_path>")
        sys.exit(1)
    generate(Path(sys.argv[1]))
