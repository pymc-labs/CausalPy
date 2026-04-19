"""Minimal demo of all ``plot_xY`` kinds with synthetic posterior data.

Writes one PNG (2×2 grid) under ``scripts/plot_xY_demo_output/`` (gitignored).

Run::

    python scripts/plot_xY_visualization_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from causalpy.constants import HDI_PROB
from causalpy.plot_utils import plot_xY

OUTPUT_DIR = Path(__file__).resolve().parent / "plot_xY_demo_output"


def main() -> None:
    rng = np.random.default_rng(42)
    n_chains, n_draws, n_time = 2, 200, 40
    x = pd.date_range("2022-01-01", periods=n_time, freq="D")
    trend = 10 + 0.05 * np.arange(n_time) + 0.01 * np.arange(n_time) ** 2
    draw_means = trend + rng.normal(0, 0.4, (n_chains, n_draws, n_time))
    samples = draw_means + rng.normal(0, 0.8, (n_chains, n_draws, n_time))

    y = xr.DataArray(
        samples,
        dims=["chain", "draw", "obs_ind"],
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "obs_ind": x,
        },
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    plot_xY(
        x,
        y,
        ax=axes[0, 0],
        kind="ribbon",
        ci_kind="hdi",
        ci_prob=HDI_PROB,
        plot_hdi_kwargs={"color": "C0", "fill_kwargs": {"alpha": 0.3}},
        label="HDI",
    )
    axes[0, 0].set_title("Ribbon (HDI)")
    axes[0, 0].set_ylabel("Value")

    plot_xY(
        x,
        y,
        ax=axes[0, 1],
        kind="ribbon",
        ci_kind="eti",
        ci_prob=0.89,
        plot_hdi_kwargs={"color": "C1", "fill_kwargs": {"alpha": 0.3}},
        label="ETI",
    )
    axes[0, 1].set_title("Ribbon (ETI)")

    plot_xY(
        x,
        y,
        ax=axes[1, 0],
        kind="histogram",
        plot_hdi_kwargs={"cmap": "Blues", "alpha": 0.85, "color": "white"},
        label="Posterior",
    )
    axes[1, 0].set_title("Histogram (2D heatmap)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Value")

    plot_xY(
        x,
        y,
        ax=axes[1, 1],
        kind="spaghetti",
        num_samples=60,
        plot_hdi_kwargs={"color": "C3", "alpha": 0.12},
        label="Samples",
    )
    axes[1, 1].set_title("Spaghetti (60 draws)")
    axes[1, 1].set_xlabel("Time")

    for ax in axes.ravel():
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle("plot_xY: all kinds (synthetic posterior)", fontsize=12, y=1.02)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "plot_xY_all_kinds_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
