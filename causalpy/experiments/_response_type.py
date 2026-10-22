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
"""Shared wiring for expectation versus posterior-predictive uncertainty.

Stored experiment results stay on the conditional expectation ``mu``. Plot
and summary callers opt into observation-noise draws (``y_hat``) without
mutating that stored bundle.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
from matplotlib.figure import Figure

ResponseType = Literal["expectation", "prediction"]
Quantity = Literal["mu", "y_hat"]

logger = logging.getLogger(__name__)

_logged = {"emitted": False}

_RESPONSE_LOG = (
    "response_type selects the uncertainty quantity. "
    "'expectation' (default) uses the conditional mean mu and excludes "
    "observation noise. 'prediction' uses posterior predictive y_hat. "
    "Where that series includes observation noise, intervals are wider. "
    "Backends whose y_hat aliases mu reject prediction mode. "
    "Stored impacts remain the conditional expectation."
)

_PREDICTION_SUMMARY_NOTE = (
    " These intervals include observation noise (a posterior predictive "
    "contrast of y_hat), not the default conditional-expectation impact."
)


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


def reset_response_type_log() -> None:
    """Clear the once-per-process plot and summary log flag. For tests."""
    _logged["emitted"] = False


def log_response_type_once(where: Literal["plot", "summary"]) -> None:
    """Explain the response-type choice once per process.

    The first plot or summary call consumes the flag. A later call site
    does not log again.

    Parameters
    ----------
    where : {"plot", "summary"}
        Call site. Both values share one process-wide flag.
    """
    if where not in ("plot", "summary"):
        raise ValueError(f"where must be 'plot' or 'summary', got {where!r}")
    if _logged["emitted"]:
        return
    _logged["emitted"] = True
    logger.info(_RESPONSE_LOG)


def _shifted_replay_seed(seed: Any, offset: int) -> Any | None:
    """Return a different seed when PyMC would replay ``seed``, else None.

    ``int``, numpy integers, and one-dimensional integer sequences are
    replayed by ``sample_posterior_predictive``. ``None`` and a
    ``Generator`` are not, so they need no offset.
    """
    if isinstance(seed, bool):
        return None
    if isinstance(seed, (int, np.integer)):
        return int(seed) + offset
    values = None
    if isinstance(seed, np.ndarray):
        values = seed
    elif isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
        values = np.asarray(seed)
    if values is None or values.dtype.kind not in "iu" or values.size == 0:
        return None
    if values.ndim == 0:
        return int(values) + offset
    if values.ndim != 1:
        return None
    return (values.astype(int) + offset).tolist()


@contextmanager
def independent_predict_seed(model: Any, offset: int = 1) -> Iterator[None]:
    """Shift a seeded model's predictive seed for one call, then restore it.

    ``PyMCModel.predict`` reseeds posterior-predictive sampling from
    ``sample_kwargs["random_seed"]`` on every call. Two equal-length
    designs therefore share one observation-noise sequence, and a
    fitted-minus-counterfactual contrast cancels that noise. Integer
    seeds and chain-length integer sequences are shifted for the
    duration of the context. Missing seeds and generators are left alone.

    Parameters
    ----------
    model : object
        Object with ``sample_kwargs["random_seed"]``. Other objects are ignored.
    offset : int, default 1
        Added to a replayed seed while the context is active.

    Yields
    ------
    None
        Control returns with the seed shifted. The original seed is restored
        on exit, including when the body raises.
    """
    kwargs = getattr(model, "sample_kwargs", None)
    seed = kwargs.get("random_seed") if isinstance(kwargs, dict) else None
    shifted = _shifted_replay_seed(seed, offset)
    if shifted is None:
        yield
        return
    kwargs["random_seed"] = shifted
    try:
        yield
    finally:
        kwargs["random_seed"] = seed


def annotation_text(
    *,
    response_type: str,
    ci_prob: float,
    ci_kind: str,
    kind: str,
    has_draws: bool,
) -> str:
    """Figure note naming the drawn quantity and, for ribbons, the interval.

    Parameters
    ----------
    response_type : str
        ``"expectation"`` or ``"prediction"``.
    ci_prob : float
        Credible-interval probability, used only for ribbon notes.
    ci_kind : str
        ``"hdi"`` or ``"eti"``. The note uses the selected label.
    kind : str
        Plot kind. Only ``"ribbon"`` includes an interval percentage.
    has_draws : bool
        ``False`` names a point prediction instead of an interval.

    Returns
    -------
    str
        The figure note.
    """
    if not has_draws:
        return "Point prediction (no observation-noise interval)."
    checked = check_response_type(response_type)
    if checked == "expectation":
        label = "conditional expectation (excludes observation noise)"
    else:
        label = "posterior predictive (includes observation noise)"
    if kind != "ribbon":
        return f"Drawn uncertainty: {label}"
    interval = "ETI" if ci_kind == "eti" else "HDI"
    percent = f"{ci_prob * 100:g}%"
    return f"Shaded band: {percent} {interval} of the {label}"


def should_annotate(
    *,
    show_response_annotation: bool | None,
    kind: str,
    has_draws: bool,
) -> bool:
    """Whether a figure note should be added.

    Parameters
    ----------
    show_response_annotation : bool or None
        ``None`` annotates Bayesian ribbon plots. ``True`` always annotates.
        ``False`` never does.
    kind : str
        Plot kind. The default annotates only ``"ribbon"``.
    has_draws : bool
        Whether the plotted quantity has more than one draw.

    Returns
    -------
    bool
        Whether to add the figure note.
    """
    if show_response_annotation is False:
        return False
    if show_response_annotation is True:
        return True
    return has_draws and kind == "ribbon"


def annotate_response(
    fig: Figure,
    *,
    response_type: str,
    ci_prob: float,
    ci_kind: str,
    kind: str,
    show_response_annotation: bool | None,
    has_draws: bool,
) -> None:
    """Add the response-type note when the call asks for it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to annotate. The note is figure text, not an axis title.
    response_type : str
        ``"expectation"`` or ``"prediction"``.
    ci_prob : float
        Credible-interval probability, used only for ribbon notes.
    ci_kind : str
        ``"hdi"`` or ``"eti"``.
    kind : str
        Plot kind.
    show_response_annotation : bool or None
        See :func:`should_annotate`.
    has_draws : bool
        Whether the plotted quantity has more than one draw.
    """
    if not should_annotate(
        show_response_annotation=show_response_annotation,
        kind=kind,
        has_draws=has_draws,
    ):
        return
    # Figure coordinates, not subplots_adjust: experiment plots already called
    # tight_layout, and adjusting afterwards warns and can be ignored.
    fig.text(
        0.01,
        0.01,
        annotation_text(
            response_type=response_type,
            ci_prob=ci_prob,
            ci_kind=ci_kind,
            kind=kind,
            has_draws=has_draws,
        ),
        ha="left",
        va="bottom",
        fontsize=8,
    )


def finish_response_plot(
    fig: Figure,
    axes: Any,
    *,
    response_type: str,
    ci_prob: float,
    ci_kind: str,
    kind: str,
    show_response_annotation: bool | None,
    has_draws: bool,
) -> tuple[Figure, Any]:
    """Log the quantity choice once and optionally label the figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to annotate.
    axes : Any
        Axes returned with the figure, unchanged.
    response_type : str
        ``"expectation"`` or ``"prediction"``.
    ci_prob : float
        Credible-interval probability, used only for ribbon notes.
    ci_kind : str
        ``"hdi"`` or ``"eti"``.
    kind : str
        Plot kind.
    show_response_annotation : bool or None
        See :func:`should_annotate`.
    has_draws : bool
        Whether the plotted quantity has more than one draw. Logging is
        skipped for point estimates.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, Any)
        The same figure and axes, after the optional note.
    """
    if has_draws:
        log_response_type_once("plot")
    annotate_response(
        fig,
        response_type=response_type,
        ci_prob=ci_prob,
        ci_kind=ci_kind,
        kind=kind,
        show_response_annotation=show_response_annotation,
        has_draws=has_draws,
    )
    return fig, axes


def note_prediction_summary(text: str, response_type: str) -> str:
    """Append an observation-noise warning when a summary used ``y_hat``.

    Parameters
    ----------
    text : str
        Summary prose. Returned unchanged for ``"expectation"``.
    response_type : str
        ``"expectation"`` or ``"prediction"``.

    Returns
    -------
    str
        Summary text, with the observation-noise note when required.
    """
    if check_response_type(response_type) != "prediction":
        return text
    if _PREDICTION_SUMMARY_NOTE.strip() in text:
        return text
    return text.rstrip() + _PREDICTION_SUMMARY_NOTE
