#   Copyright 2022 - 2026 The PyMC Labs Developers
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
"""DataTree / ArviZ migration contract tests."""

from __future__ import annotations

import subprocess
import sys


def test_cold_import_causalpy_without_arviz_migration_warning() -> None:
    """``import causalpy`` in a fresh process must not raise ``arviz.MigrationWarning``.

    ``conftest`` already imports CausalPy in-process, so this check has to run in a
    cold subprocess. The subprocess imports ArviZ, installs a category-specific
    warnings-as-errors filter, then imports CausalPy.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import warnings\n"
                "import arviz\n"
                "warnings.filterwarnings("
                "'error', category=arviz.MigrationWarning)\n"
                "import causalpy\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
