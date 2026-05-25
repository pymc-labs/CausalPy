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
"""CausalPy agent skills for AI-assisted causal inference.

User-facing skills that teach AI agents how to use CausalPy.  Ship with the
pip wheel so the installed version always matches the library API.

Developer skills live in ``.github/skills/`` and are auto-discovered
in-repo via platform symlinks; they are **not** included here.
"""

from causalpy.skills._installer import install, list_skills, uninstall

__all__ = ["install", "list_skills", "uninstall"]
