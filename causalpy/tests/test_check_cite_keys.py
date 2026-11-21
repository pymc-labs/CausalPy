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
"""Tests for the docs citation key check."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check_cite_keys.py"


def _load_checker_module():
    spec = importlib.util.spec_from_file_location("check_cite_keys", CHECK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def docs_source(tmp_path):
    """A minimal docs tree with one bibliography holding ``smith2020``."""
    (tmp_path / "conf.py").write_text('bibtex_bibfiles = ["refs.bib"]\n')
    (tmp_path / "refs.bib").write_text(
        "@comment{ignored,}\n"
        "@string{jasa = {Journal}}\n"
        "@article{Smith2020,\n  title = {A},\n}\n"
        "@book{jones2019, title = {B}}\n"
    )
    return tmp_path


def _notebook(markdown: str, code: str = "") -> str:
    return json.dumps(
        {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": [markdown]},
                {"cell_type": "code", "metadata": {}, "source": code, "outputs": []},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def test_known_keys_pass_in_every_syntax(docs_source):
    """Known keys pass in MyST and RST roles, lists, and any case."""
    (docs_source / "page.md").write_text(
        "See {cite:p}`smith2020` and {cite:t}`jones2019, SMITH2020` and {footcite}`jones2019`."
    )
    (docs_source / "glossary.rst").write_text(":footcite:t:`smith2020`\n")
    (docs_source / "nb.ipynb").write_text(_notebook("{cite:p}`jones2019`"))
    checker = _load_checker_module()
    assert checker.find_unknown_citations(docs_source) == []


@pytest.mark.parametrize(
    ("name", "content"),
    [
        pytest.param("page.md", "See {cite:p}`smith2021`.", id="myst-md"),
        pytest.param("page.md", "{cite:t}`smith2020,smith2021`", id="myst-list"),
        pytest.param("glossary.rst", ":footcite:p:`smith2021`\n", id="rst-role"),
        pytest.param("nb.ipynb", _notebook("{cite:t}`smith2021`"), id="notebook"),
    ],
)
def test_unknown_key_is_reported(docs_source, name, content):
    """A cited key that is not in the bibliography is reported with its file."""
    (docs_source / name).write_text(content)
    checker = _load_checker_module()
    assert checker.find_unknown_citations(docs_source) == [
        (docs_source / name, "smith2021")
    ]


def test_bib_comment_and_string_entries_are_not_keys(docs_source):
    """``@comment`` and ``@string`` blocks do not define citable keys."""
    (docs_source / "page.md").write_text("{cite:p}`ignored` {cite:p}`jasa`")
    checker = _load_checker_module()
    assert [key for _, key in checker.find_unknown_citations(docs_source)] == [
        "ignored",
        "jasa",
    ]


def test_notebook_code_cells_and_build_dir_are_ignored(docs_source):
    """Only notebook markdown cells and source files count, not code or build output."""
    (docs_source / "nb.ipynb").write_text(
        _notebook("no citations here", code="print('{cite:p}`smith2021`')")
    )
    (docs_source / "_build").mkdir()
    (docs_source / "_build" / "page.md").write_text("{cite:p}`smith2021`")
    checker = _load_checker_module()
    assert checker.find_unknown_citations(docs_source) == []


def test_cite_key_check_cli_passes_on_repo_docs():
    """Every citation in the real docs resolves."""
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout
