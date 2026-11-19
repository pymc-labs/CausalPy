"""Simulate and fit the homepage ITS example using CausalPy's public API."""

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

import causalpy as cp

# Outcome units; slope is outcome units per month.
truth = {
    "intercept": 45.0,
    "slope": 0.35,
    "seasonality": 7.0,
    "effect": 18.0,
    "sigma": 1.8,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("data") / "interrupted-time-series.json",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(sum(map(ord, "interrupted time series animation")))
    t = np.arange(72)
    split = 48
    season = np.sin(2 * np.pi * t / 12)
    untreated = truth["intercept"] + truth["slope"] * t + truth["seasonality"] * season
    noise = rng.normal(0, truth["sigma"], len(t))
    effect = np.where(t >= split, truth["effect"], 0)
    df = pd.DataFrame(
        {"t": t, "season": season, "y": untreated + effect + noise},
        index=pd.date_range("2020-01-01", periods=len(t), freq="MS"),
    )
    formula = "y ~ 1 + t + season"
    experiment = cp.InterruptedTimeSeries(
        df,
        pd.Timestamp("2024-01-01"),
        formula=formula,
        model=cp.pymc_models.LinearRegression(),
    ).fit()
    plot = experiment.get_plot_data(hdi_prob=0.94)
    payload = {
        "dates": plot.index.strftime("%b %Y").tolist(),
        "observed": plot.y.tolist(),
        "split": split,
        "truth": truth,
        "hdi_prob": 0.94,
        "metadata": {
            "interval_target": "expectation",
            "interval_probability": 0.94,
            "formula": formula,
            "backend": "causalpy.pymc_models.LinearRegression",
            "causalpy_version": cp.__version__,
            "note": "Synthetic observations; prediction and impact HDIs from get_plot_data(). Predictive intervals await issue #295.",
        },
    }
    for key, column in {
        "prediction": "prediction",
        "impact": "impact",
        "pred_lower": "pred_hdi_lower_94",
        "pred_upper": "pred_hdi_upper_94",
        "impact_lower": "impact_hdi_lower_94",
        "impact_upper": "impact_hdi_upper_94",
    }.items():
        values = plot[column].to_numpy()
        assert np.isfinite(values).all(), column
        payload[key] = values.tolist()
    for prefix in ("pred", "impact"):
        assert np.all(
            np.array(payload[prefix + "_lower"]) < np.array(payload[prefix + "_upper"])
        )
    assert len(experiment.datapre) == split
    assert np.allclose(plot.impact, plot.y - plot.prediction)
    print(az.summary(experiment.model.idata, var_names=["beta", "y_hat_sigma"]))
    print(
        "Divergences:", int(experiment.model.idata["sample_stats"]["diverging"].sum())
    )
    print(
        f"True effect: {truth['effect']}; average impact: {plot.impact.iloc[split:].mean():.2f}"
    )
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
