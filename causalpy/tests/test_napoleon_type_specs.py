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
import builtins
import importlib
import importlib.util
import inspect
import re
import typing
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
_NAPOLEON_SECTIONS = _napoleon_types._NAPOLEON_SECTIONS
_TYPE_SECTIONS = _napoleon_types._TYPE_SECTIONS
_BARE_SECTIONS = frozenset(
    {"return", "returns", "yield", "yields", "raise", "raises", "warn", "warns"}
)
_SCANNED_SECTIONS = _TYPE_SECTIONS | {"raise", "raises"}


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


def test_returns_section_does_not_spill_into_examples() -> None:
    lines = [
        "Returns",
        "-------",
        "list[int]",
        "",
        "Examples",
        "--------",
        "list[matplotlib.axes.Axes]",
    ]
    normalize_docstring_lines(None, "method", "probe", None, None, lines)
    assert lines[2] == "list of int"
    assert lines[-1] == "list[matplotlib.axes.Axes]"


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


_ROLE = re.compile(r":(?:py:)?(?:class|obj|data|func|meth|type|exc):`~?([^`]+)`")
# Names checked against the live intersphinx inventories. A new external alias
# has to be added here only after that inventory actually contains it.
_INVENTORY_TARGETS = frozenset(
    {
        "True",
        "False",
        # arviz.InferenceData is literalized: the live inventory does not list it.
        "callable",
        "collections.abc.Mapping",
        "collections.abc.Sequence",
        "matplotlib.axes.Axes",
        "matplotlib.figure.Figure",
        "numpy.inf",
        "numpy.ndarray",
        "pandas.DataFrame",
        "pandas.DatetimeIndex",
        "pandas.Index",
        "pandas.Series",
        "pandas.Timestamp",
        "pandas.api.extensions.ExtensionArray",
        "pathlib.Path",
        "pymc.model.core.Deterministic",
        "pymc.model.core.Model",
        "pymc_extras.prior.Prior",
        "pytensor.tensor.TensorVariable",
        "sklearn.base.RegressorMixin",
        "typing.Any",
        "typing.Self",
        "xarray.DataArray",
        "xarray.DataTree",
        "xarray.Dataset",
    }
)


def _alias_targets() -> set[str]:
    return {
        match.group(1)
        for value in TYPE_ALIASES.values()
        if (match := _ROLE.search(value)) is not None
    }


def _public_type_specs() -> list[str]:
    specs: list[str] = []
    for path in (_ROOT / "causalpy").rglob("*.py"):
        parts = path.relative_to(_ROOT).parts
        if any(part in {"tests", "data"} or part.startswith("_") for part in parts):
            continue
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            docstring = ast.get_docstring(node)
            if not docstring:
                continue
            lines = docstring.splitlines()
            normalize_docstring_lines(
                None, "member", getattr(node, "name", ""), None, None, lines
            )
            specs.extend(_field_specs(lines))
    return specs


def _field_specs(lines: list[str]) -> list[str]:
    specs: list[str] = []
    in_section = False
    bare_type = False
    field_indent: int | None = None
    for index, raw in enumerate(lines):
        if _is_section_header(lines, index):
            header = raw.strip().lower().strip(":")
            in_section = header in _SCANNED_SECTIONS
            bare_type = header in _BARE_SECTIONS
            field_indent = None
            continue
        if not in_section or not raw.strip() or _is_underline(raw):
            if (
                in_section
                and not raw.strip()
                and index + 1 < len(lines)
                and not lines[index + 1].strip()
            ):
                in_section = False
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if field_indent is None:
            field_indent = indent
        if indent < field_indent:
            in_section = False
            continue
        if indent != field_indent or raw.strip().startswith(":"):
            continue
        text = raw.strip()
        spec = text.split(":", 1)[1].strip() if ":" in text else text
        if not spec or (":" not in text and not bare_type):
            continue
        if any(role in spec for role in (":class:", ":func:", ":obj:", ":data:")):
            continue
        specs.append(spec)
    return specs


def _is_section_header(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    name = lines[index].strip().lower().strip(":")
    underline = lines[index + 1].strip()
    return (
        name in _NAPOLEON_SECTIONS
        and bool(underline)
        and len(set(underline)) == 1
        and underline[0] in "-="
        and len(underline) >= len(lines[index].strip())
    )


def _is_underline(raw: str) -> bool:
    rule = raw.strip()
    return bool(rule) and len(set(rule)) == 1 and rule[0] in "-="


def test_internal_alias_targets_are_public() -> None:
    for target in sorted(_alias_targets()):
        if not target.startswith("causalpy."):
            continue
        module_name, attr = target.rsplit(".", 1)
        assert not any(part.startswith("_") for part in module_name.split("."))
        assert not attr.startswith("_")
        module = importlib.import_module(module_name)
        assert hasattr(module, attr), target


def test_external_alias_targets_are_in_known_inventories() -> None:
    external = {
        target for target in _alias_targets() if not target.startswith("causalpy.")
    }
    assert external <= _INVENTORY_TARGETS


def test_promoted_type_tokens_are_aliased_or_builtin() -> None:
    pytest.importorskip("sphinx")
    from sphinx.ext.napoleon import Config
    from sphinx.ext.napoleon.docstring import NumpyDocstring

    allowed = _alias_targets()
    allowed.update(
        name for name in dir(builtins) if inspect.isclass(getattr(builtins, name))
    )
    allowed.update(typing.__all__)
    allowed.add("None")
    config = Config(
        napoleon_preprocess_types=True,
        napoleon_type_aliases=TYPE_ALIASES,
        napoleon_use_rtype=True,
    )
    unresolved: dict[str, str] = {}
    for spec in dict.fromkeys(_public_type_specs()):
        rendered = "\n".join(
            NumpyDocstring(
                ["Parameters", "----------", f"x : {spec}", "    unused"],
                config,
            ).lines()
        )
        for match in _ROLE.finditer(rendered):
            target = match.group(1)
            if target not in allowed:
                unresolved.setdefault(target, spec)
    assert unresolved == {}


def test_inference_data_alias_is_a_literal() -> None:
    target = TYPE_ALIASES["InferenceData"]
    assert target == "``arviz.InferenceData``"
    assert ":class:" not in target


def test_preprocess_gate_follows_the_defining_module() -> None:
    pymc = pytest.importorskip("pymc")
    from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries

    assert _napoleon_types.preprocess_types_for(pymc.Model.check_start_vals) is False
    assert _napoleon_types.preprocess_types_for(InterruptedTimeSeries.plot) is True


def test_inherited_pymc_raises_are_not_turned_into_roles() -> None:
    pytest.importorskip("sphinx")
    pymc = pytest.importorskip("pymc")
    from sphinx.ext.napoleon import Config

    lines = inspect.getdoc(pymc.Model.check_start_vals).splitlines()
    app = type("App", (), {"config": Config(napoleon_preprocess_types=True)})()
    _napoleon_types.process_owned_docstring(
        app,
        "method",
        "causalpy.InterruptedTimeSeries.check_start_vals",
        pymc.Model.check_start_vals,
        None,
        lines,
    )
    rendered = "\n".join(lines)
    assert ":py:class:`the parameters" not in rendered
    assert ":py:class:`an invalid" not in rendered
    assert "the parameters" in rendered


def test_owned_docstrings_still_link_matplotlib_returns() -> None:
    pytest.importorskip("sphinx")
    from sphinx.ext.napoleon import Config

    from causalpy.experiments.interrupted_time_series import InterruptedTimeSeries

    lines = inspect.cleandoc(InterruptedTimeSeries.plot.__doc__).splitlines()
    normalize_docstring_lines(
        None,
        "method",
        "causalpy.experiments.interrupted_time_series.InterruptedTimeSeries.plot",
        InterruptedTimeSeries.plot,
        None,
        lines,
    )
    app = type(
        "App",
        (),
        {
            "config": Config(
                napoleon_preprocess_types=True,
                napoleon_type_aliases=TYPE_ALIASES,
                napoleon_use_rtype=False,
            )
        },
    )()
    _napoleon_types.process_owned_docstring(
        app,
        "method",
        "causalpy.experiments.interrupted_time_series.InterruptedTimeSeries.plot",
        InterruptedTimeSeries.plot,
        None,
        lines,
    )
    rendered = "\n".join(lines)
    assert ":class:`~matplotlib.figure.Figure`" in rendered
    assert ":class:`~matplotlib.axes.Axes`" in rendered


def test_install_replaces_the_napoleon_processor() -> None:
    pytest.importorskip("sphinx")
    from types import SimpleNamespace

    from sphinx.ext.napoleon import _process_docstring

    class _Events:
        def __init__(self) -> None:
            self.listeners = {
                "autodoc-process-docstring": [
                    SimpleNamespace(id=7, handler=_process_docstring, priority=500)
                ]
            }
            self.connected: list[tuple[str, object, int]] = []

        def disconnect(self, listener_id: int) -> None:
            assert listener_id == 7
            self.listeners["autodoc-process-docstring"].clear()

        def connect(self, name: str, callback: object, priority: int = 500) -> int:
            self.connected.append((name, callback, priority))
            return 8

    events = _Events()
    _napoleon_types.install_preprocess_gate(SimpleNamespace(events=events))
    assert events.listeners["autodoc-process-docstring"] == []
    assert events.connected == [
        ("autodoc-process-docstring", _napoleon_types.process_owned_docstring, 500)
    ]
