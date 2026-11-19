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
"""Select and label expectation or observation-level predictive outcomes."""

from typing import Any, Literal

from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

ResponseType = Literal["expectation", "prediction"]
Quantity = Literal["mu", "y_hat"]
_NOTE_FONT_SIZE = 8


def check_response_type(response_type: str) -> ResponseType:
    """Return ``response_type`` or raise if it is not a known value.

    Parameters
    ----------
    response_type : str
        ``"expectation"`` or ``"prediction"``.

    Returns
    -------
    {"expectation", "prediction"}
        The validated response type.
    """
    if response_type not in ("expectation", "prediction"):
        raise ValueError(
            "response_type must be 'expectation' or 'prediction', "
            f"got {response_type!r}"
        )
    return response_type  # type: ignore[return-value]


def quantity_for(response_type: str) -> Quantity:
    """Map a response type to the model variable it draws.

    Parameters
    ----------
    response_type : str
        ``"expectation"`` selects ``mu``; ``"prediction"`` selects ``y_hat``.

    Returns
    -------
    {"mu", "y_hat"}
        The prediction-container variable for that response type.
    """
    checked = check_response_type(response_type)
    return "mu" if checked == "expectation" else "y_hat"


def _wrap_response_note(note: str, width_inches: float) -> str:
    """Wrap fixed-size notes using actual font widths in physical points."""
    font = FontProperties(size=_NOTE_FONT_SIZE)
    metrics = TextToPath()
    lines: list[str] = []
    line = ""
    for word in note.split():
        candidate = f"{line} {word}".strip()
        width, _, _ = metrics.get_text_width_height_descent(candidate, font, False)
        if line and width > width_inches * 72:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    return "\n".join(lines)


def annotation_text(
    *,
    response_type: str,
    group: Literal["prior", "posterior"] = "posterior",
    target: str = "outcome",
    ci_prob: float,
    ci_kind: str,
    kind: str,
    has_draws: bool,
    show_response_annotation: bool | None = True,
    width_inches: float | None = None,
) -> str | None:
    """Figure note naming the drawn quantity and, for ribbons, the interval.

    Parameters
    ----------
    target : str
        Predictive quantity named in the note: outcome or difference.
    group : {"prior", "posterior"}
        Conditioning draws used for the displayed outcomes.
    response_type : str
        ``"expectation"`` or ``"prediction"``.
    ci_prob : float
        Credible-interval probability, used only for ribbon notes.
    ci_kind : str
        ``"hdi"`` or ``"eti"``. The note uses the selected label.
    kind : str
        Plot kind. Only ``"ribbon"`` includes an interval percentage.
    show_response_annotation : bool or None
        ``None`` labels draw-based ribbons; ``True`` always labels and
        ``False`` suppresses the note.
    width_inches : float, optional
        Available note width. Wrap using the fixed note font when provided.
    has_draws : bool
        ``False`` names a point prediction instead of an interval.

    Returns
    -------
    str or None
        The figure note, or ``None`` when no annotation was requested.
    """
    if show_response_annotation is False or (
        show_response_annotation is None and (not has_draws or kind != "ribbon")
    ):
        return None
    if not has_draws:
        note = "Point prediction (no observation-noise interval)."
    else:
        checked = check_response_type(response_type)
        if checked == "expectation":
            label = (
                "expected difference"
                if target == "difference"
                else "conditional expectation"
            )
            label = f"{group} {label} (excludes observation noise)"
        else:
            label = f"{group} predictive {target} (includes observation noise)"
        subject = "Difference" if target == "difference" else "Outcome"
        if kind != "ribbon":
            note = f"{subject} uncertainty: {label}"
        else:
            interval = "ETI" if ci_kind == "eti" else "HDI"
            percent = f"{ci_prob * 100:g}%"
            note = f"{subject} intervals: {percent} {interval} of the {label}"
    return note if width_inches is None else _wrap_response_note(note, width_inches)


def finish_response_plot(
    fig: Figure,
    axes: Any,
    *,
    response_type: str,
    group: Literal["prior", "posterior"] = "posterior",
    target: str = "outcome",
    ci_prob: float,
    ci_kind: str,
    kind: str,
    show_response_annotation: bool | None,
    has_draws: bool,
) -> tuple[Figure, Any]:
    """Optionally label outcome uncertainty without changing effect estimands.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to annotate.
    axes : Any
        Axes returned with the figure, unchanged.
    target : str
        Predictive quantity named in the note: outcome or difference.
    group : {"prior", "posterior"}
        Conditioning draws used for the displayed outcomes.
    response_type : str
        ``"expectation"`` or ``"prediction"``.
    ci_prob : float
        Credible-interval probability, used only for ribbon notes.
    ci_kind : str
        ``"hdi"`` or ``"eti"``.
    kind : str
        Plot kind.
    show_response_annotation : bool or None
        ``None`` labels draw-based ribbons; ``True`` always labels and
        ``False`` suppresses the note.
    has_draws : bool
        Whether the plotted quantity has more than one draw.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, Any)
        The same figure and axes, after the optional note.
    """
    note = annotation_text(
        response_type=response_type,
        group=group,
        target=target,
        ci_prob=ci_prob,
        ci_kind=ci_kind,
        kind=kind,
        has_draws=has_draws,
        show_response_annotation=show_response_annotation,
        width_inches=fig.get_figwidth() * 0.98,
    )
    if note is None:
        return fig, axes
    artist = fig.text(
        0.01, 0.01, note, ha="left", va="bottom", fontsize=_NOTE_FONT_SIZE
    )
    # Reserve the wrapped note's actual height before laying out panel labels.
    bounds = artist.get_window_extent().transformed(fig.transFigure.inverted())
    fig.set_layout_engine("tight", rect=(0, bounds.y1 + 0.015, 1, 1))
    return fig, axes
