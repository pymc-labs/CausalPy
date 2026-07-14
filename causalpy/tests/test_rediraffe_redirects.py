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
"""Tests for rediraffe redirect maintenance."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check_rediraffe_redirects.py"
CONF_PY = REPO_ROOT / "docs" / "source" / "conf.py"


def _load_checker_module():
    spec = importlib.util.spec_from_file_location(
        "check_rediraffe_redirects", CHECK_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rediraffe_redirect_targets_exist() -> None:
    checker = _load_checker_module()
    redirects = checker.load_redirects()
    for _src, dst in redirects.items():
        assert checker.target_exists(dst), dst


def test_rediraffe_redirect_check_passes_against_main() -> None:
    checker = _load_checker_module()
    errors = checker.check_redirects(None)
    assert errors == [], "\n".join(errors)


def test_rediraffe_redirect_check_cli_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
