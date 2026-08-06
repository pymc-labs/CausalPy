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
"""Configurable plot themes for CausalPy experiment figures.

Historically :meth:`~causalpy.experiments.base.BaseExperiment._render_plot`
hard-coded the ``arviz-darkgrid`` matplotlib style. This module turns that into
a small, opt-in registry so the same experiment ``plot()`` methods can render in
different visual themes without any per-plot changes.

The default theme is unchanged (``arviz-darkgrid``), so existing figures,
notebooks, and user code look exactly as before. An ``editorial`` theme provides
a light, publication-oriented (NYT/FT-inspired) look used by the rendered plots
in the API docstrings.

A theme bundles three things:

* **rcParams** applied for the duration of the draw call (background, spines,
  grid, colour cycle, tick styling, ...).
* **Semantic colours** (:class:`PlotColors`) for the few marks that are *not*
  driven by the matplotlib colour cycle — the difference-in-differences causal
  impact arrow and the zero/treatment reference rules. These let those accents
  follow the active theme instead of being hard-coded.
* An optional **title font** stack applied to axis titles after drawing, so a
  theme can use a serif headline over sans-serif data labels.

The public surface is intentionally tiny: :func:`plot_style` (a context manager),
:func:`set_plot_style` (a global switch), and the registry helpers. Naming is
kept theme-centric so the registry can later map to declarative
:mod:`plotnine` themes (see issue #988) without changing callers.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import arviz as az
import matplotlib as mpl
import matplotlib.colors as mcolors
from cycler import cycler
from matplotlib.lines import Line2D

# -- Editorial palette (PyMC brand) ------------------------------------------
# Colours anchor on the official PyMC brand palette so figures read as native to
# the PyMC ecosystem. Source: https://github.com/pymc-devs/brand (the brand repo
# ships these as swatch PNGs, e.g. ``colorswatch_12698a_green.png`` for the teal
# and ``colorswatch_504a4e_black.png`` for the charcoal). The near-white ground
# is an editorial choice (PyMC's own ground is white ``#ffffff``); everything
# else is brand teal + charcoal. Exposed as module constants so tests and
# downstream code can reference the exact values.
PYMC_TEAL = "#12698a"  #: PyMC primary brand colour (logo teal / petrol blue).
PYMC_CHARCOAL = "#504a4e"  #: PyMC brand dark (logo wordmark charcoal).

EDITORIAL_BG = "#faf9f7"  #: Near-white warm background (editorial ground).
EDITORIAL_INK = PYMC_CHARCOAL  #: Text, observed points, reference rules, accents.
EDITORIAL_MUTED = "#7d7377"  #: Muted charcoal tint for labels and reference rules.
EDITORIAL_SPINE = "#9a9296"  #: Colour of the single (bottom) axis spine.
EDITORIAL_GRID = "#dcd6ca"  #: Subtle horizontal gridline colour.

#: Ordered qualitative colour cycle mapped onto the ``C0``/``C1``/``C2`` roles
#: the experiment ``_plot`` methods use (``C0`` = control / pre-period, ``C1`` =
#: treated / counterfactual). Led by the two PyMC brand colours, followed by
#: teal/charcoal-family tints for any additional series.
EDITORIAL_CYCLE = [
    PYMC_TEAL,  # C0 PyMC teal
    PYMC_CHARCOAL,  # C1 PyMC charcoal
    "#5aa0b8",  # C2 light teal
    "#8a8388",  # C3 warm grey
    "#0e4e68",  # C4 deep teal
    "#b7c9d1",  # C5 pale teal-grey
]

# NOTE on the font choice (issue #391). The editorial theme sets a *serif title
# over sans-serif data labels* — the NYT/FT headline convention. We deliberately
# use only serif faces that ship *with matplotlib* (STIX Two Text, STIXGeneral,
# DejaVu Serif) rather than a more distinctive system or web font. The docstring
# plots are rendered on Read the Docs, which installs via pip (not the conda
# env), so any face that is not bundled with matplotlib — e.g. Charter, Georgia,
# Ubuntu, or the NYT-analog "Libre Franklin" — would silently fall back to the
# default and render differently on RTD than it does locally. Shipping a
# distinctive sans (Libre Franklin is OFL-licensed) as a bundled repo asset
# registered through matplotlib.font_manager is a possible future upgrade; it is
# intentionally left out here to keep this change dependency-free and its output
# reproducible across environments.
#: Serif title font stack; first available wins, ``DejaVu Serif`` is the floor.
EDITORIAL_TITLE_FONT: tuple[str, ...] = ("STIX Two Text", "STIXGeneral", "DejaVu Serif")

EDITORIAL_RCPARAMS: dict[str, object] = {
    "figure.facecolor": EDITORIAL_BG,
    "axes.facecolor": EDITORIAL_BG,
    "savefig.facecolor": EDITORIAL_BG,
    "savefig.edgecolor": EDITORIAL_BG,
    "font.family": "sans-serif",
    "text.color": EDITORIAL_INK,
    "axes.edgecolor": EDITORIAL_SPINE,
    "axes.linewidth": 1.1,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": EDITORIAL_GRID,
    "grid.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": True,
    "axes.titlelocation": "left",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.titlecolor": EDITORIAL_INK,
    "axes.labelcolor": EDITORIAL_MUTED,
    "axes.labelsize": 11,
    "xtick.color": EDITORIAL_MUTED,
    "ytick.color": EDITORIAL_MUTED,
    "xtick.labelcolor": EDITORIAL_MUTED,
    "ytick.labelcolor": EDITORIAL_MUTED,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 7,
    "ytick.major.pad": 7,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "axes.prop_cycle": cycler(color=EDITORIAL_CYCLE),
}


@dataclass(frozen=True)
class PlotColors:
    """Theme-driven colours for marks that are not on the matplotlib colour cycle.

    Attributes
    ----------
    impact : str
        Colour for emphasis marks such as the difference-in-differences causal
        impact arrow.
    reference : str
        Colour for reference rules such as the zero line on impact panels and
        the treatment-time marker.
    """

    impact: str
    reference: str


@dataclass(frozen=True)
class PlotTheme:
    """A named plotting theme.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``"editorial"``.
    rcparams : dict or None
        Matplotlib rcParams applied while drawing. ``None`` means "use the
        matplotlib/ArviZ style named ``name``" (how the default
        ``arviz-darkgrid`` theme defers to :data:`arviz.style.library`).
    colors : PlotColors
        Semantic accent colours (see :class:`PlotColors`).
    title_font : tuple of str or None
        Font family stack applied to axis titles after drawing, or ``None`` to
        leave titles in the rcParams font.
    """

    name: str
    rcparams: dict[str, object] | None
    colors: PlotColors
    title_font: tuple[str, ...] | None


_THEMES: dict[str, PlotTheme] = {
    # Default: byte-for-byte the previous behaviour (green arrow, black rules,
    # arviz-darkgrid style, rcParams title font).
    "arviz-darkgrid": PlotTheme(
        name="arviz-darkgrid",
        rcparams=None,
        colors=PlotColors(impact="green", reference="k"),
        title_font=None,
    ),
    "editorial": PlotTheme(
        name="editorial",
        rcparams=EDITORIAL_RCPARAMS,
        colors=PlotColors(impact=EDITORIAL_INK, reference=EDITORIAL_MUTED),
        title_font=EDITORIAL_TITLE_FONT,
    ),
}

#: Name of the active theme. Defaults to ``"arviz-darkgrid"`` to preserve the
#: historical look for all existing callers.
_ACTIVE: str = "arviz-darkgrid"


def available_styles() -> list[str]:
    """Return the names of the registered themes."""
    return list(_THEMES)


def get_theme(name: str) -> PlotTheme:
    """Return the :class:`PlotTheme` registered under ``name``.

    Parameters
    ----------
    name : str
        A registered theme name (see :func:`available_styles`).

    Raises
    ------
    ValueError
        If ``name`` is not a registered theme.
    """
    try:
        return _THEMES[name]
    except KeyError:
        raise ValueError(
            f"Unknown plot style {name!r}. Available: {available_styles()}."
        ) from None


def active_theme() -> PlotTheme:
    """Return the currently active :class:`PlotTheme`."""
    return _THEMES[_ACTIVE]


def active_colors() -> PlotColors:
    """Return the active theme's semantic accent colours.

    Experiment ``_plot`` methods call this for marks that are not driven by the
    matplotlib colour cycle, so those accents follow the active theme.
    """
    return active_theme().colors


def style_context(theme: PlotTheme | None = None) -> list:
    """Build the argument for :func:`matplotlib.pyplot.style.context`.

    Parameters
    ----------
    theme : PlotTheme, optional
        Theme to render. Defaults to the :func:`active_theme`.

    Returns
    -------
    list
        A style spec list suitable for ``plt.style.context(...)``. Custom
        rcParams themes are composed on top of the matplotlib defaults so the
        result is independent of any ambient global style; the default
        ``arviz-darkgrid`` theme defers to :data:`arviz.style.library` exactly
        as before.
    """
    theme = theme or active_theme()
    if theme.rcparams is None:
        return [az.style.library[theme.name]]
    return ["default", theme.rcparams]


def apply_title_font(fig: mpl.figure.Figure, fonts: Sequence[str]) -> None:
    """Apply a font family stack to every axis title (and suptitle) on ``fig``.

    Lets a theme use a serif headline over sans-serif data labels without each
    ``_plot`` method having to opt in.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure whose titles to restyle.
    fonts : sequence of str
        Font family stack (first available wins).
    """
    family = list(fonts)
    for ax in fig.axes:
        title = ax.title
        if title is not None and title.get_text():
            title.set_fontfamily(family)
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text():
        suptitle.set_fontfamily(family)


_CN_RE = re.compile(r"^C\d+$")


def _is_cycle_ref(color: object) -> bool:
    """True for a matplotlib cycle reference string like ``"C0"``/``"C1"``."""
    return isinstance(color, str) and bool(_CN_RE.match(color))


def bake_cycle_colors(fig: mpl.figure.Figure) -> None:
    """Freeze deferred ``"CN"`` cycle colours to concrete RGBA on ``fig``.

    matplotlib resolves ``"C0"``/``"C1"``/... against
    ``rcParams["axes.prop_cycle"]`` **lazily, at draw time**, for
    :class:`~matplotlib.lines.Line2D`. A figure built inside a temporary style
    context but *drawn later* by the caller — the API-docstring ``.. plot::``
    directive, a notebook cell, or a bare ``fig.savefig`` — would otherwise
    re-resolve those references against whatever cycle is active at draw time
    (usually the matplotlib default), silently discarding the theme's colours.

    Calling this **inside** the theme's style context resolves each reference
    against the active (themed) cycle and writes the concrete colour back onto
    the artist, so the returned figure keeps the theme's colours wherever it is
    later drawn. Only lines need this: fill/band collections resolve their
    colours eagerly at creation, so they already carry concrete RGBA.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure whose artists to freeze. Mutated in place.
    """
    for line in fig.findobj(Line2D):
        for getter, setter in (
            ("get_color", "set_color"),
            ("get_markerfacecolor", "set_markerfacecolor"),
            ("get_markeredgecolor", "set_markeredgecolor"),
        ):
            color = getattr(line, getter)()
            if _is_cycle_ref(color):
                getattr(line, setter)(mcolors.to_rgba(color))


def set_plot_style(name: str) -> str:
    """Set the active plot theme globally and return the previous theme name.

    Parameters
    ----------
    name : str
        A registered theme name (see :func:`available_styles`).

    Returns
    -------
    str
        The name of the theme that was active before this call, so it can be
        restored.

    Examples
    --------
    >>> import causalpy as cp
    >>> previous = cp.set_plot_style("editorial")
    >>> _ = cp.set_plot_style(previous)  # restore
    """
    global _ACTIVE
    get_theme(name)  # validate
    previous, _ACTIVE = _ACTIVE, name
    return previous


@contextmanager
def plot_style(name: str) -> Iterator[PlotTheme]:
    """Temporarily activate a plot theme.

    Parameters
    ----------
    name : str
        A registered theme name (see :func:`available_styles`).

    Yields
    ------
    PlotTheme
        The activated theme.

    Examples
    --------
    >>> import causalpy as cp
    >>> df = cp.load_data("did")
    >>> result = cp.DifferenceInDifferences(
    ...     df,
    ...     formula="y ~ 1 + group*post_treatment",
    ...     time_variable_name="t",
    ...     group_variable_name="group",
    ...     model=cp.pymc_models.LinearRegression(
    ...         sample_kwargs={
    ...             "draws": 500,
    ...             "tune": 500,
    ...             "chains": 2,
    ...             "random_seed": 42,
    ...             "progressbar": False,
    ...         }
    ...     ),
    ... )
    >>> with cp.plot_style("editorial"):
    ...     fig, ax = result.plot()
    """
    previous = set_plot_style(name)
    try:
        yield active_theme()
    finally:
        set_plot_style(previous)
