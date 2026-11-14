"""Make Napoleon type fields resolvable without rewriting source docstrings.

Sphinx 9 tokenizes a type on every comma, including commas inside brackets and
parentheses, before it looks the token up in ``napoleon_type_aliases``. A field
such as ``list[matplotlib.axes.Axes]`` is therefore one missing class, not a
``list`` of ``Axes``. This module rewrites those fields, on the lines Napoleon
is about to parse, into the ``list of ...`` / ``A or B`` forms the tokenizer
already links. Aliases name the object Sphinx documents; a token with no such
object stays a literal so preprocessing does not emit a missing reference.
Preprocessing stays off for inherited third-party docstrings. Their Raises
blocks are not NumPy fields, and Napoleon would turn the prose into missing
class roles. Napoleon also writes those role strings into
``autodoc_type_aliases`` while looking up an untyped field. The config view
hands it a copy, so a later ``get_type_hints`` call does not evaluate a role
as an annotation. The checked-in docstrings stay in the typing style the code uses.
"""

from __future__ import annotations

import re
from typing import Any

_TYPE_SECTIONS = frozenset(
    {
        "args",
        "arguments",
        "attributes",
        "keyword args",
        "keyword arguments",
        "other parameters",
        "parameters",
        "receive",
        "receives",
        "return",
        "returns",
        "warn",
        "warns",
        "yield",
        "yields",
    }
)
# Napoleon ends a field section at any of these headers, not only type
# sections. Without this, a Returns block rewrites the following Examples.
_NAPOLEON_SECTIONS = _TYPE_SECTIONS | frozenset(
    {
        "attention",
        "caution",
        "danger",
        "error",
        "example",
        "examples",
        "hint",
        "important",
        "methods",
        "note",
        "notes",
        "raise",
        "raises",
        "references",
        "see also",
        "tip",
        "todo",
        "warning",
        "warnings",
    }
)
_BARE_TYPE_SECTIONS = frozenset({"return", "returns", "yield", "yields"})
_UNDERLINE = re.compile(r"^[=\-`:.'\"~^_*+#<>]{2,}\s*$")
_FIELD = re.compile(
    r"^(?P<indent>[ ]*)"
    r"(?P<name>\*{0,2}[A-Za-z_][\w.]*(\s*,\s*\*{0,2}[A-Za-z_][\w.]*)*)"
    r"\s*:\s*(?P<type>.+)$"
)
_XREF = re.compile(r":(?:[a-zA-Z][\w.+-]*:)?`")
_GENERIC = re.compile(r"(?P<head>[A-Za-z_][\w.]*)\[(?P<inner>[^\[\]]*)\]")
_TRAILING_KIND = re.compile(
    r"^(?P<kind>default|shape)\s*[=:]?\s*(?P<value>.*)$", re.IGNORECASE
)
_OF_HEADS = frozenset(
    {
        "Collection",
        "Deque",
        "FrozenSet",
        "Iterable",
        "List",
        "Sequence",
        "Set",
        "Tuple",
        "collection",
        "deque",
        "frozenset",
        "iterable",
        "list",
        "sequence",
        "set",
        "tuple",
    }
)
_DICT_HEADS = frozenset(
    {"Dict", "Mapping", "OrderedDict", "defaultdict", "dict", "mapping"}
)
_LOWER_HEADS = {
    "Dict": "dict",
    "FrozenSet": "frozenset",
    "List": "list",
    "Set": "set",
    "Tuple": "tuple",
    "Type": "type",
}


def _class(target: str) -> str:
    return f":class:`~{target}`"


def _data(target: str) -> str:
    return f":data:`~{target}`"


def _func(target: str) -> str:
    return f":func:`~{target}`"


def _literal(text: str) -> str:
    return f"``{text}``"


def _alias(*names: str, target: str) -> dict[str, str]:
    return dict.fromkeys(names, target)


# Fully qualified names are what the docstrings already write. Short and
# ``np.``/``pd.``/``plt.`` forms cover the same objects when a field uses them.
# A literal alias is a token Napoleon would otherwise emit as a missing class.
TYPE_ALIASES: dict[str, str] = {
    **_alias(
        "matplotlib.figure.Figure",
        "Figure",
        "plt.Figure",
        target=_class("matplotlib.figure.Figure"),
    ),
    **_alias(
        "matplotlib.axes.Axes",
        "Axes",
        "plt.Axes",
        target=_class("matplotlib.axes.Axes"),
    ),
    **_alias("numpy.ndarray", "ndarray", "np.ndarray", target=_class("numpy.ndarray")),
    **_alias(
        "pandas.DataFrame",
        "DataFrame",
        "pd.DataFrame",
        target=_class("pandas.DataFrame"),
    ),
    **_alias("pandas.Series", "Series", "pd.Series", target=_class("pandas.Series")),
    **_alias("pandas.Index", "Index", "pd.Index", target=_class("pandas.Index")),
    **_alias(
        "pandas.DatetimeIndex",
        "DatetimeIndex",
        "pd.DatetimeIndex",
        target=_class("pandas.DatetimeIndex"),
    ),
    **_alias(
        "pandas.Timestamp",
        "Timestamp",
        "pd.Timestamp",
        target=_class("pandas.Timestamp"),
    ),
    **_alias(
        "pandas.api.extensions.ExtensionArray",
        "ExtensionArray",
        target=_class("pandas.api.extensions.ExtensionArray"),
    ),
    **_alias(
        "xarray.DataArray",
        "DataArray",
        "xr.DataArray",
        target=_class("xarray.DataArray"),
    ),
    **_alias(
        "xarray.Dataset", "Dataset", "xr.Dataset", target=_class("xarray.Dataset")
    ),
    **_alias(
        "xarray.DataTree",
        "DataTree",
        "xr.DataTree",
        target=_class("xarray.DataTree"),
    ),
    # The ArviZ HTML page exists, but objects.inv does not list InferenceData.
    # A class role would be a new missing target.
    **_alias(
        "arviz.InferenceData",
        "InferenceData",
        "az.InferenceData",
        target=_literal("arviz.InferenceData"),
    ),
    **_alias(
        "pymc.Model",
        "Model",
        "pm.Model",
        target=_class("pymc.model.core.Model"),
    ),
    **_alias(
        "pytensor.tensor.TensorVariable",
        "TensorVariable",
        "pt.TensorVariable",
        target=_class("pytensor.tensor.TensorVariable"),
    ),
    # Not in the pytensor inventory. A class role would be a new missing target.
    **_alias(
        "pytensor.tensor.random.RandomVariable",
        "RandomVariable",
        target=_literal("pytensor.tensor.random.RandomVariable"),
    ),
    **_alias(
        "sklearn.base.RegressorMixin",
        "RegressorMixin",
        target=_class("sklearn.base.RegressorMixin"),
    ),
    **_alias(
        "pymc_extras.prior.Prior",
        "Prior",
        target=_class("pymc_extras.prior.Prior"),
    ),
    **_alias("pathlib.Path", "Path", target=_class("pathlib.Path")),
    **_alias(
        "collections.abc.Mapping",
        "Mapping",
        "mapping",
        target=_class("collections.abc.Mapping"),
    ),
    **_alias(
        "collections.abc.Sequence",
        "Sequence",
        "sequence",
        target=_class("collections.abc.Sequence"),
    ),
    **_alias("typing.Any", "Any", target=":obj:`~typing.Any`"),
    "callable": ":py:func:`callable`",
    "Self": ":py:data:`~typing.Self`",
    "True": ":py:data:`True`",
    "False": ":py:data:`False`",
    "pm.Deterministic": _func("pymc.model.core.Deterministic"),
    "np.inf": _data("numpy.inf"),
    "HDI_PROB": _data("causalpy.constants.HDI_PROB"),
    # Package-root names only resolve when ``causalpy.__all__`` exports them.
    # Everything else must use the module autosummary actually documents.
    **_alias(
        "BaseExperiment",
        target=_class("causalpy.experiments.base.BaseExperiment"),
    ),
    **_alias("EffectSummary", target=_class("causalpy.EffectSummary")),
    **_alias("CheckResult", target=_class("causalpy.checks.base.CheckResult")),
    **_alias("Check", target=_class("causalpy.checks.base.Check")),
    **_alias("Pipeline", target=_class("causalpy.Pipeline")),
    **_alias("PipelineContext", target=_class("causalpy.PipelineContext")),
    **_alias("PipelineResult", target=_class("causalpy.PipelineResult")),
    **_alias("Step", target=_class("causalpy.Step")),
    **_alias("PyMCModel", target=_class("causalpy.pymc_models.PyMCModel")),
    **_alias(
        "PyMCForecastModel",
        target=_class("causalpy.pymc_forecast_models.PyMCForecastModel"),
    ),
    **_alias(
        "InstrumentalVariableRegression",
        target=_class("causalpy.pymc_models.InstrumentalVariableRegression"),
    ),
    **_alias(
        "PropensityScore",
        target=_class("causalpy.pymc_models.PropensityScore"),
    ),
    **_alias(
        "AssuranceResult",
        target=_class("causalpy.checks.operating_characteristics.AssuranceResult"),
    ),
    **_alias(
        "OperatingCharacteristics",
        target=_class(
            "causalpy.checks.operating_characteristics.OperatingCharacteristics"
        ),
    ),
    **_alias(
        "ModelAdapter",
        target=_class("causalpy.experiments.model_adapter.ModelAdapter"),
    ),
    **_alias(
        "AugmentedSyntheticControl",
        target=_class(
            "causalpy.experiments.augmented_synthetic_control.AugmentedSyntheticControl"
        ),
    ),
    **_alias(
        "InteractiveFixedEffectsSyntheticControl",
        target=_class(
            "causalpy.experiments.interactive_fixed_effects."
            "InteractiveFixedEffectsSyntheticControl"
        ),
    ),
    **_alias(
        "InteractiveFixedEffectsResult",
        target=_class(
            "causalpy.experiments.interactive_fixed_effects.InteractiveFixedEffectsResult"
        ),
    ),
    **_alias(
        "SensitivitySummary",
        target=_class("causalpy.steps.sensitivity.SensitivitySummary"),
    ),
    **_alias(
        "SyntheticControl",
        target=_class("causalpy.experiments.synthetic_control.SyntheticControl"),
    ),
    **_alias("BadIndexException", target=_class("causalpy.BadIndexException")),
    **_alias("DataException", target=_class("causalpy.DataException")),
    **_alias(
        "GroupNotSampledException",
        target=_class("causalpy.GroupNotSampledException"),
    ),
    **_alias(
        "PriorPredictiveNotSupportedException",
        target=_class("causalpy.PriorPredictiveNotSupportedException"),
    ),
    **_alias("SensitivityAnalysis", target=_class("causalpy.SensitivityAnalysis")),
    "DataFrameLike": ":py:type:`~causalpy.input_data.DataFrameLike`",
    "StaggeredDifferenceInDifferencesResult": _literal(
        "StaggeredDifferenceInDifferencesResult"
    ),
    "SyntheticDifferenceInDifferencesResult": _literal(
        "SyntheticDifferenceInDifferencesResult"
    ),
    "_ScalarBayesianStats": _literal("_ScalarBayesianStats"),
    # Private, and not a documented pymc-forecast object.
    "_NullModelUnidentifiedError": _literal("_NullModelUnidentifiedError"),
    "pymc_forecast.ForecastingModel": _literal("pymc_forecast.ForecastingModel"),
    "array-like": _literal("array-like"),
    "dataframe-like": _literal("dataframe-like"),
    "array": _literal("array"),
    "distribution": _literal("distribution"),
    "function": _literal("function"),
    "plotnine.ggplot": _literal("plotnine.ggplot"),
    "plotnine composition": _literal("plotnine composition"),
    "PyMC variable": _literal("PyMC variable"),
    "shape": _literal("shape"),
}


def normalize_docstring_lines(
    _app: object,
    _what: str,
    _name: str,
    _obj: object,
    _options: object,
    lines: list[str],
) -> None:
    """Rewrite type fields in place before Napoleon tokenizes them."""
    index = 0
    while index < len(lines):
        header = lines[index].strip().lower().strip(":")
        underline = lines[index + 1] if index + 1 < len(lines) else ""
        if header not in _TYPE_SECTIONS or not _is_underline(lines[index], underline):
            index += 1
            continue
        bare_types = header in _BARE_TYPE_SECTIONS
        index += 2
        field_indent: int | None = None
        while index < len(lines):
            if _section_ends(lines, index, field_indent):
                break
            raw = lines[index]
            if not raw.strip():
                index += 1
                continue
            indent = _indent_of(raw)
            if field_indent is None:
                field_indent = indent
            if indent != field_indent:
                index += 1
                continue
            rewritten, note = _rewrite_field(raw, bare_type=bare_types)
            if rewritten != raw:
                lines[index] = rewritten
            if note and not _description_has(lines, index, field_indent, note):
                lines.insert(index + 1, " " * (field_indent + 4) + note)
                index += 1
            index += 1


def preprocess_types_for(obj: object) -> bool:
    """Return whether Napoleon may turn this object's type tokens into roles.

    Inherited methods keep the defining module, so a documented name under
    ``causalpy`` is not enough. PyMC's ``check_start_vals`` is the case this
    rejects: preprocessing wraps ``the parameters`` in a class role.
    """
    module = _defining_module(obj)
    return module == "causalpy" or module.startswith("causalpy.")


def _defining_module(obj: object) -> str:
    module = getattr(obj, "__module__", None)
    if isinstance(module, str) and module:
        return module
    wrapped = getattr(obj, "__func__", None)
    if wrapped is None:
        wrapped = getattr(obj, "fget", None)
    module = getattr(wrapped, "__module__", None)
    return module if isinstance(module, str) else ""


class _NapoleonConfig:
    """Config view Napoleon can mutate without poisoning autodoc.

    ``_lookup_annotation`` updates ``autodoc_type_aliases`` in place with
    ``napoleon_type_aliases``. Those values are docstring roles, not
    annotations. The copy keeps that update off the dict autodoc later passes
    to ``get_type_hints``.
    """

    def __init__(self, config: object, *, preprocess_types: bool | None = None) -> None:
        self._config = config
        self._preprocess_types = preprocess_types

    @property
    def napoleon_preprocess_types(self) -> bool:
        if self._preprocess_types is False:
            return False
        return bool(getattr(self._config, "napoleon_preprocess_types", False))

    @property
    def autodoc_type_aliases(self) -> dict[str, str]:
        raw = getattr(self._config, "autodoc_type_aliases", {}) or {}
        return dict(raw)

    def __getattr__(self, name: str) -> object:
        return getattr(self._config, name)


def process_owned_docstring(
    app: Any,
    what: str,
    name: str,
    obj: object,
    options: object,
    lines: list[str],
) -> None:
    """Run Napoleon, but do not preprocess types outside ``causalpy``."""
    from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring

    config = _NapoleonConfig(
        app.config,
        preprocess_types=False if not preprocess_types_for(obj) else None,
    )
    result = lines
    if config.napoleon_numpy_docstring:
        result = NumpyDocstring(result, config, app, what, name, obj, options).lines()
    if config.napoleon_google_docstring:
        result = GoogleDocstring(result, config, app, what, name, obj, options).lines()
    lines[:] = list(result)


def install_preprocess_gate(app: object) -> None:
    """Replace Napoleon's processor without moving its listener slot."""
    from sphinx.events import EventListener
    from sphinx.ext.napoleon import _process_docstring

    listeners = app.events.listeners.get("autodoc-process-docstring", [])
    replaced = False
    for index, listener in enumerate(listeners):
        if listener.handler is _process_docstring:
            # disconnect+connect appends after other priority-500 handlers,
            # including sphinx-autodoc-typehints when it still uses the default.
            listeners[index] = EventListener(
                listener.id, process_owned_docstring, listener.priority
            )
            replaced = True
    if not replaced:
        msg = "sphinx.ext.napoleon docstring processor is not connected"
        raise RuntimeError(msg)


def rewrite_type_spec(spec: str) -> tuple[str, str]:
    """Return a Napoleon-safe type string and a note for any displaced clause."""
    kept, notes = _peel_unsafe_clauses(spec.strip())
    rewritten = _rewrite_pipes(_rewrite_tuple_parens(_rewrite_brackets(kept)))
    return rewritten, " ".join(notes)


def _is_underline(header: str, underline: str) -> bool:
    rule = underline.strip()
    return (
        bool(rule)
        and bool(_UNDERLINE.fullmatch(rule))
        and len(rule) >= len(header.strip())
    )


def _section_ends(lines: list[str], index: int, field_indent: int | None) -> bool:
    raw = lines[index]
    if not raw.strip():
        return index + 1 < len(lines) and not lines[index + 1].strip()
    if _is_header_at(lines, index):
        return True
    return field_indent is not None and _indent_of(raw) < field_indent


def _is_header_at(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    name = lines[index].strip().lower().strip(":")
    return name in _NAPOLEON_SECTIONS and _is_underline(lines[index], lines[index + 1])


def _rewrite_field(raw: str, *, bare_type: bool) -> tuple[str, str]:
    if _XREF.search(raw):
        return raw, ""
    match = _FIELD.match(raw)
    if match:
        spec, note = rewrite_type_spec(match.group("type"))
        if spec == match.group("type").strip() and not note:
            return raw, ""
        indent = match.group("indent")
        name = match.group("name")
        colon = " : " if " : " in raw else ": "
        return f"{indent}{name}{colon}{spec}", note
    if not bare_type:
        return raw, ""
    spec, note = rewrite_type_spec(raw.strip())
    if spec == raw.strip() and not note:
        return raw, ""
    return f"{' ' * _indent_of(raw)}{spec}", note


def _description_has(
    lines: list[str], index: int, field_indent: int, note: str
) -> bool:
    parts = note.split("``")
    payload = parts[1] if len(parts) >= 3 else note
    for raw in lines[index + 1 :]:
        if not raw.strip():
            continue
        if _indent_of(raw) <= field_indent:
            break
        if payload and payload in raw:
            return True
    return False


def _peel_unsafe_clauses(spec: str) -> tuple[str, list[str]]:
    clauses = [part.strip() for part in _split_top_level(spec, ",") if part.strip()]
    if len(clauses) < 2:
        return spec, []
    kept = [clauses[0]]
    notes: list[str] = []
    for clause in clauses[1:]:
        if _needs_displacement(clause):
            notes.append(_note_for(clause))
        else:
            kept.append(clause)
    return ", ".join(kept), notes


def _needs_displacement(clause: str) -> bool:
    return _has_comma_outside_braces(clause)


def _note_for(clause: str) -> str:
    match = _TRAILING_KIND.match(clause.strip())
    if match is None:
        return f"Also ``{clause.strip()}``."
    kind = match.group("kind").lower()
    value = match.group("value").strip()
    if kind == "default":
        return f"Defaults to ``{value}``."
    return f"Shape ``{value}``."


def _rewrite_brackets(text: str) -> str:
    while True:
        match = _next_generic(text)
        if match is None:
            return text
        text = (
            text[: match.start()]
            + _format_generic(match.group("head"), match.group("inner"))
            + text[match.end() :]
        )


def _next_generic(text: str) -> re.Match[str] | None:
    for match in _GENERIC.finditer(text):
        if not _in_literal(text, match.start()):
            return match
    return None


def _in_literal(text: str, pos: int) -> bool:
    quote: str | None = None
    backtick = False
    for index, char in enumerate(text[:pos]):
        if quote is not None:
            if char == quote and text[index - 1] != "\\":
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "`":
            backtick = not backtick
    return quote is not None or backtick


def _format_generic(head: str, inner: str) -> str:
    args = [part.strip() for part in _split_top_level(inner, ",") if part.strip()]
    key = head.rsplit(".", 1)[-1]
    shown = _LOWER_HEADS.get(key, key)
    if key in {"Optional"} and len(args) == 1:
        return f"{args[0]} or None"
    if key in {"Union"}:
        return " or ".join(args)
    if key in {"Literal"}:
        return "{" + ", ".join(args) + "}"
    if key in _DICT_HEADS and len(args) >= 2:
        label = "dict" if shown == "dict" else shown
        return f"{label} of {args[0]} to {' and '.join(args[1:])}"
    if key in {"tuple", "Tuple"} and args:
        if args[-1] == "...":
            return f"tuple of {args[0]}" if len(args) == 2 else "tuple"
        return "tuple of " + " and ".join(args)
    if (key in _OF_HEADS or key in {"type", "Type"}) and len(args) == 1:
        return f"{shown} of {args[0]}"
    if len(args) == 1:
        return f"{shown} of {args[0]}"
    if args:
        return f"{shown} of " + " and ".join(args)
    return head


def _rewrite_tuple_parens(text: str) -> str:
    needle = "tuple of ("
    cursor = 0
    parts: list[str] = []
    while True:
        found = text.find(needle, cursor)
        if found < 0 or not _at_depth_zero(text, found):
            parts.append(text[cursor:])
            break
        open_at = found + len("tuple of ")
        close_at = _matching(text, open_at, "(", ")")
        if close_at < 0:
            parts.append(text[cursor:])
            break
        args = [
            part.strip()
            for part in _split_top_level(text[open_at + 1 : close_at], ",")
            if part.strip()
        ]
        replacement = "tuple of " + (" and ".join(args) if len(args) != 1 else args[0])
        parts.append(text[cursor:found])
        parts.append(replacement)
        cursor = close_at + 1
    rewritten = "".join(parts)
    if rewritten == text:
        return text
    return _rewrite_tuple_parens(rewritten)


def _rewrite_pipes(text: str) -> str:
    parts = [part.strip() for part in _split_top_level(text, "|") if part.strip()]
    if len(parts) < 2:
        return text
    return " or ".join(parts)


def _split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depths = {"(": 0, "[": 0, "{": 0}
    quote: str | None = None
    backtick = False
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            buf.append(char)
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            buf.append(char)
            index += 1
            continue
        if char == "`":
            backtick = not backtick
            buf.append(char)
            index += 1
            continue
        if not backtick:
            if char in depths:
                depths[char] += 1
            elif char in {")", "]", "}"}:
                opener = {")": "(", "]": "[", "}": "{"}[char]
                if depths[opener]:
                    depths[opener] -= 1
            elif all(depth == 0 for depth in depths.values()) and text.startswith(
                sep, index
            ):
                parts.append("".join(buf))
                buf = []
                index += len(sep)
                continue
        buf.append(char)
        index += 1
    parts.append("".join(buf))
    return parts


def _has_comma_outside_braces(text: str) -> bool:
    brace = 0
    quote: str | None = None
    backtick = False
    for index, char in enumerate(text):
        if quote is not None:
            if char == quote and text[index - 1] != "\\":
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "`":
            backtick = not backtick
            continue
        if backtick:
            continue
        if char == "{":
            brace += 1
        elif char == "}" and brace:
            brace -= 1
        elif char == "," and brace == 0:
            return True
    return False


def _at_depth_zero(text: str, pos: int) -> bool:
    depths = {"(": 0, "[": 0, "{": 0}
    quote: str | None = None
    backtick = False
    for index, char in enumerate(text[:pos]):
        if quote is not None:
            if char == quote and text[index - 1] != "\\":
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "`":
            backtick = not backtick
            continue
        if backtick:
            continue
        if char in depths:
            depths[char] += 1
        elif char in {")", "]", "}"}:
            opener = {")": "(", "]": "[", "}": "{"}[char]
            if depths[opener]:
                depths[opener] -= 1
    return (
        quote is None and not backtick and all(depth == 0 for depth in depths.values())
    )


def _matching(text: str, open_at: int, opener: str, closer: str) -> int:
    depth = 0
    quote: str | None = None
    for index in range(open_at, len(text)):
        char = text[index]
        if quote is not None:
            if char == quote and text[index - 1] != "\\":
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))
