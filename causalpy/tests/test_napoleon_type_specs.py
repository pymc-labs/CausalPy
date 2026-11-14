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
"""Napoleon cannot link a type it splits on an inner comma.

These tests lock the render-time rewrite that turns ``list[matplotlib.axes.Axes]``
into tokens Sphinx can resolve, including the interrupted-time-series plot page
from issue 901.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _ROOT / "docs" / "source" / "napoleon_types.py"
_SPEC = importlib.util.spec_from_file_location("napoleon_types", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_napoleon_types = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_napoleon_types)

rewrite_type_spec = _napoleon_types.rewrite_type_spec
normalize_docstring_lines = _napoleon_types.normalize_docstring_lines
TYPE_ALIASES = _napoleon_types.TYPE_ALIASES


def _type_of(line: str) -> str:
    lines = ["Parameters", "----------", line, "    unused"]
    normalize_docstring_lines(None, "method", "probe", None, None, lines)
    return lines[2].split(":", 1)[1].strip()


@pytest.mark.parametrize(
    ("spec", "expected", "note_prefix"),
    [
        ("list[matplotlib.axes.Axes]", "list of matplotlib.axes.Axes", ""),
        ("tuple of (float, float)", "tuple of float and float", ""),
        ("tuple of float, default (7, 8)", "tuple of float", "Defaults to"),
        (
            "array-like, shape (n_samples, n_covariates)",
            "array-like",
            "Shape",
        ),
        (
            "list[dict[str, str]], optional",
            "list of dict of str to str, optional",
            "",
        ),
        (
            "Union[int, float, pd.Timestamp]",
            "int or float or pd.Timestamp",
            "",
        ),
        ("set[str] | None, default None", "set of str or None, default None", ""),
        (
            '{"prior", "posterior"}, default "posterior"',
            '{"prior", "posterior"}, default "posterior"',
            "",
        ),
        ("xr.DataTree | None", "xr.DataTree or None", ""),
        ("Optional[Any]", "Any or None", ""),
        ("Mapping[str, CheckResult]", "Mapping of str to CheckResult", ""),
        ("type[RegressorMixin]", "type of RegressorMixin", ""),
        ("matplotlib.figure.Figure", "matplotlib.figure.Figure", ""),
        ("tuple[plt.Figure, plt.Axes]", "tuple of plt.Figure and plt.Axes", ""),
    ],
)
def test_rewrite_type_spec(spec: str, expected: str, note_prefix: str) -> None:
    rewritten, note = rewrite_type_spec(spec)
    assert rewritten == expected
    if note_prefix:
        assert note.startswith(note_prefix)
    else:
        assert note == ""
    assert rewrite_type_spec(rewritten) == (rewritten, "")


def test_default_already_described_is_not_repeated() -> None:
    lines = [
        "Parameters",
        "----------",
        "figsize : tuple of float, default (7, 8)",
        "    Width and height. Defaults to ``(7, 8)``.",
    ]
    normalize_docstring_lines(None, "method", "probe", None, None, lines)
    assert lines[2] == "figsize : tuple of float"
    assert lines[3:] == ["    Width and height. Defaults to ``(7, 8)``."]


def test_examples_section_is_left_alone() -> None:
    lines = [
        "Examples",
        "--------",
        "list[matplotlib.axes.Axes]",
    ]
    original = list(lines)
    normalize_docstring_lines(None, "method", "probe", None, None, lines)
    assert lines == original


def test_interrupted_time_series_plot_returns_are_linkable() -> None:
    source = (
        _ROOT / "causalpy" / "experiments" / "interrupted_time_series.py"
    ).read_text()
    module = ast.parse(source)
    plot = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "plot"
    )
    docstring = ast.get_docstring(plot, clean=True)
    assert docstring is not None
    lines = docstring.splitlines()
    normalize_docstring_lines(
        None, "method", "InterruptedTimeSeries.plot", None, None, lines
    )
    rendered = "\n".join(lines)
    assert "fig : matplotlib.figure.Figure" in rendered
    assert "ax : list of matplotlib.axes.Axes" in rendered
    assert "list[matplotlib.axes.Axes]" not in rendered
    assert "tuple of (float, float)" not in rendered


def test_package_type_fields_have_no_comma_generics() -> None:
    leftovers: list[str] = []
    for path in (_ROOT / "causalpy").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(
                node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                continue
            docstring = ast.get_docstring(node, clean=True)
            if not docstring:
                continue
            lines = docstring.splitlines()
            normalize_docstring_lines(
                None, "member", getattr(node, "name", ""), None, None, lines
            )
            leftovers.extend(
                _broken_type_fields(path, getattr(node, "name", ""), lines)
            )
    assert leftovers == []


def _broken_type_fields(path: Path, name: str, lines: list[str]) -> list[str]:
    broken: list[str] = []
    in_section = False
    field_indent: int | None = None
    for index, raw in enumerate(lines):
        header = raw.strip().lower().strip(":")
        underline = lines[index + 1] if index + 1 < len(lines) else ""
        if (
            header
            in {
                "parameters",
                "returns",
                "yields",
                "attributes",
                "other parameters",
            }
            and underline.strip()
            and len(set(underline.strip())) == 1
        ):
            in_section = True
            field_indent = None
            continue
        if not in_section or not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if field_indent is None:
            field_indent = indent
        if indent < field_indent or (
            index + 1 < len(lines)
            and raw.strip().lower() in {"examples", "notes", "see also"}
        ):
            in_section = False
            continue
        if (
            indent != field_indent
            or ":" not in raw
            or ":class:" in raw
            or ":func:" in raw
        ):
            continue
        spec = raw.split(":", 1)[1]
        if "[" in spec or "tuple of (" in spec or " | " in spec:
            broken.append(f"{path.relative_to(_ROOT)}:{name}: {raw.strip()}")
    return broken


def test_napoleon_links_the_plot_return_types() -> None:
    pytest.importorskip("sphinx")
    from sphinx.ext.napoleon import Config
    from sphinx.ext.napoleon.docstring import NumpyDocstring

    source = (
        _ROOT / "causalpy" / "experiments" / "interrupted_time_series.py"
    ).read_text()
    module = ast.parse(source)
    plot = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "plot"
    )
    docstring = ast.get_docstring(plot, clean=True)
    assert docstring is not None
    lines = docstring.splitlines()
    normalize_docstring_lines(
        None, "method", "InterruptedTimeSeries.plot", None, None, lines
    )
    rendered = "\n".join(
        NumpyDocstring(
            lines,
            Config(
                napoleon_preprocess_types=True,
                napoleon_type_aliases=TYPE_ALIASES,
                napoleon_use_rtype=False,
            ),
        ).lines()
    )
    assert ":class:`~matplotlib.figure.Figure`" in rendered
    assert ":class:`~matplotlib.axes.Axes`" in rendered
    assert "list[matplotlib.axes.Axes]" not in rendered
    assert ":py:class:`list[matplotlib.axes.Axes]`" not in rendered
