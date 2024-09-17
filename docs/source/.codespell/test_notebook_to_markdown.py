#   Copyright 2024 The PyMC Labs Developers
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
"""Notebook to markdown tests."""

import os
from tempfile import TemporaryDirectory

import pytest
from notebook_to_markdown import notebook_to_markdown


@pytest.fixture
def data_dir() -> str:
    """Get current directory."""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data")


def test_notebook_to_markdown_empty_pattern(data_dir: str) -> None:
    """Test basic functionality of notebook_to_markdown with empty pattern."""
    with TemporaryDirectory() as tmp_dir:
        pattern = "*.missing"
        notebook_to_markdown(f"{data_dir}/{pattern}", tmp_dir)
        assert len(os.listdir(tmp_dir)) == 0


def test_notebook_to_markdown(data_dir: str) -> None:
    """Test basic functionality of notebook_to_markdown with a correct pattern."""
    with TemporaryDirectory() as tmp_dir:
        pattern = "*.ipynb"
        notebook_to_markdown(f"{data_dir}/{pattern}", tmp_dir)
        assert len(os.listdir(tmp_dir)) == 1
        assert "test_notebook.md" in os.listdir(tmp_dir)
