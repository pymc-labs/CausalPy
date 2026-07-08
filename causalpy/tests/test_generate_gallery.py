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
"""Tests for the example gallery generator."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SCRIPT = REPO_ROOT / "scripts" / "generate_gallery.py"
GALLERY_YAML = REPO_ROOT / "docs" / "source" / "notebooks" / "gallery.yaml"
INDEX_MD = REPO_ROOT / "docs" / "source" / "notebooks" / "index.md"


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("generate_gallery", GENERATOR_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gallery_module():
    return _load_generator_module()


def test_gallery_yaml_covers_all_notebooks_on_disk(gallery_module) -> None:
    data = yaml.safe_load(GALLERY_YAML.read_text(encoding="utf-8"))
    missing, orphan = gallery_module.check_coverage(data)
    assert missing == []
    assert orphan == []


def test_index_md_is_in_sync_with_gallery_yaml(gallery_module) -> None:
    rendered = gallery_module.render_index_md()
    on_disk = INDEX_MD.read_text(encoding="utf-8")
    assert rendered == on_disk


def test_generate_gallery_check_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "--check", "--no-thumbnails"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
