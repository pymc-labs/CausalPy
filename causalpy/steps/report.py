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
"""
GenerateReport pipeline step.

Collects effect summaries, plots, and sensitivity check results from the
pipeline context and renders a structured HTML report.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from causalpy.pipeline import PipelineContext

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


class GenerateReport:
    """Pipeline step that generates an HTML report from pipeline results.

    Parameters
    ----------
    include_plots : bool, default True
        Whether to include diagnostic plots in the report.
    include_effect_summary : bool, default True
        Whether to include the effect summary section.
    include_sensitivity : bool, default True
        Whether to include sensitivity analysis results.
    output_file : str or Path, optional
        If provided, write the HTML report to this file.

    Examples
    --------
    >>> import causalpy as cp  # doctest: +SKIP
    >>> step = cp.GenerateReport(
    ...     include_plots=True, output_file="report.html"
    ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        include_plots: bool = True,
        include_effect_summary: bool = True,
        include_sensitivity: bool = True,
        output_file: str | Path | None = None,
    ) -> None:
        self.include_plots = include_plots
        self.include_effect_summary = include_effect_summary
        self.include_sensitivity = include_sensitivity
        self.output_file = Path(output_file) if output_file else None

    def validate(self, context: PipelineContext) -> None:
        """GenerateReport has no strict prerequisites; it gracefully handles
        missing data."""

    def _render_plot(self, experiment: Any) -> list[str]:
        """Render experiment plots as base64-encoded PNG strings."""
        plots: list[str] = []
        try:
            import matplotlib.pyplot as plt

            fig, _ = experiment.plot()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            plots.append(base64.b64encode(buf.read()).decode("utf-8"))
        except Exception as exc:
            logger.debug("Could not render plot: %s", exc)
        return plots

    def run(self, context: PipelineContext) -> PipelineContext:
        """Generate the HTML report and store it in the context."""
        env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=True,
        )
        template = env.get_template("report.html")

        effect_summary = None
        effect_summary_table_html = None
        if self.include_effect_summary and context.effect_summary is not None:
            effect_summary = context.effect_summary
            if effect_summary.table is not None:
                effect_summary_table_html = effect_summary.table.to_html(
                    classes="", index=False, border=0
                )

        plots: list[str] = []
        if self.include_plots and context.experiment is not None:
            plots = self._render_plot(context.experiment)

        sensitivity_results: list[dict[str, Any]] = []
        if self.include_sensitivity and context.sensitivity_results:
            for cr in context.sensitivity_results:
                entry: dict[str, Any] = {
                    "check_name": cr.check_name,
                    "passed": cr.passed,
                    "text": cr.text,
                    "table_html": None,
                }
                if cr.table is not None:
                    entry["table_html"] = cr.table.to_html(
                        classes="", index=False, border=0
                    )
                sensitivity_results.append(entry)

        html = template.render(
            effect_summary=effect_summary,
            effect_summary_table_html=effect_summary_table_html,
            plots=plots,
            sensitivity_results=sensitivity_results,
        )

        context.report = html

        if self.output_file is not None:
            self.output_file.write_text(html, encoding="utf-8")
            logger.info("Report written to %s", self.output_file)

        return context

    def __repr__(self) -> str:
        return (
            f"GenerateReport(include_plots={self.include_plots}, "
            f"include_effect_summary={self.include_effect_summary}, "
            f"include_sensitivity={self.include_sensitivity})"
        )
