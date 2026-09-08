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
"""Drawing helpers shared by the sensitivity checks."""

from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure

# Suptitle position, just above the figure box the panels fill.
_SUPTITLE_Y = 1.02


def draw_figure(plot: Any, title: str, figsize: tuple[float, float]) -> Figure:
    """Draw a plotnine plot or composition and stamp the suptitle on it.

    ``ggplot.draw`` returns a plain matplotlib figure, which is what
    ``CheckResult.figures`` holds and what ``GenerateReport`` embeds, so
    nothing downstream needs to know a plot was built with plotnine.

    Parameters
    ----------
    plot : plotnine.ggplot or plotnine composition
        The plot to draw.
    title : str
        Figure suptitle.
    figsize : tuple of float
        Size of the drawn figure, in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The drawn figure.
    """
    figure = plot.draw()
    figure.set_size_inches(*figsize)
    # plotnine composes panels with its own layout engine, which ignores
    # subplots_adjust, so the suptitle goes above the figure box and the
    # tight bounding box used by savefig and the notebook backend grows
    # to include it.
    figure.suptitle(title, fontsize=11, fontweight="bold", y=_SUPTITLE_Y)
    return figure
