"""Batch-generate experiment plots for visual regression checking.

Fits every experiment on a small seeded dataset with tiny ``sample_kwargs``
for **both** statistical backends (PyMC and sklearn where supported), calls
the public plotting method(s), and saves each figure to
``.scratch/plot_validation/{experiment}_{backend}{suffix}_{tag}.png``.

Because everything is seeded (data generation, MCMC, matplotlib), re-running
with the same environment produces byte-identical PNGs. That makes this the
gate for plot refactors: capture a baseline before the refactor, re-run with
a new tag after, and diff the images (``cmp`` for pixel equivalence, or eyeball
the pairs when a deliberate change is expected)::

    $CONDA_EXE run -n CausalPy python scripts/generate_plots.py --tag baseline
    # ... refactor ...
    $CONDA_EXE run -n CausalPy python scripts/generate_plots.py --tag after
    for f in .scratch/plot_validation/*_baseline.png; do
        cmp "$f" "${f%_baseline.png}_after.png" && echo "OK $f"
    done

ponytail: real MCMC with tiny draws (not mocks) so plotted posterior bands are
genuine; slower but this is a validation tool, not a unit test.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

import causalpy as cp
from causalpy.data.simulate_data import (
    generate_piecewise_its_data,
    generate_staggered_did_data,
)

OUT_DIR = Path(__file__).parent.parent / ".scratch" / "plot_validation"

SAMPLE_KWARGS = {
    "tune": 100,
    "draws": 100,
    "chains": 2,
    "cores": 1,
    "progressbar": False,
    "random_seed": 42,
}


def lr():
    return cp.pymc_models.LinearRegression(sample_kwargs=SAMPLE_KWARGS)


def skl_lr():
    return SklearnLinearRegression()


# Each builder takes a backend ("pymc" or "ols") and returns
# (result, [(suffix, plot_callable), ...]). The suffix names the figure when
# an experiment produces more than one. Builders raise KeyError for
# unsupported backends, which the runner reports as SKIP.


def build_its(backend):
    df = (
        cp.load_data("its")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
    )
    r = cp.InterruptedTimeSeries(
        df,
        pd.to_datetime("2017-01-01"),
        formula="y ~ 1 + t + C(month)",
        model={"pymc": lr, "ols": skl_lr}[backend](),
    )
    return r, [("", lambda: r.plot(show=False))]


def build_sc(backend):
    model = {
        "pymc": lambda: cp.pymc_models.WeightedSumFitter(sample_kwargs=SAMPLE_KWARGS),
        "ols": lambda: cp.skl_models.WeightedProportion(),
    }[backend]()
    r = cp.SyntheticControl(
        cp.load_data("sc"),
        70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=model,
    )
    return r, [("", lambda: r.plot(show=False))]


def build_did(backend):
    r = cp.DifferenceInDifferences(
        cp.load_data("did"),
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model={"pymc": lr, "ols": skl_lr}[backend](),
    )
    return r, [("", lambda: r.plot(show=False))]


def build_piecewise_its(backend):
    df, _ = generate_piecewise_its_data(N=100, seed=42)
    r = cp.PiecewiseITS(
        df,
        formula="y ~ 1 + t + step(t, 50) + ramp(t, 50)",
        model={"pymc": lr, "ols": skl_lr}[backend](),
    )
    return r, [("", lambda: r.plot(show=False))]


def build_rd(backend):
    r = cp.RegressionDiscontinuity(
        cp.load_data("rd"),
        formula="y ~ 1 + bs(x, df=6) + treated",
        model={"pymc": lr, "ols": skl_lr}[backend](),
        treatment_threshold=0.5,
        epsilon=0.001,
    )
    return r, [("", lambda: r.plot(show=False))]


def build_staggered_did(backend):
    df = generate_staggered_did_data(
        n_units=30, n_time_periods=15, treatment_cohorts={5: 10, 10: 10}, seed=42
    )
    r = cp.StaggeredDifferenceInDifferences(
        df,
        formula="y ~ 1 + C(unit) + C(time)",
        unit_variable_name="unit",
        time_variable_name="time",
        treated_variable_name="treated",
        treatment_time_variable_name="treatment_time",
        model={"pymc": lr, "ols": skl_lr}[backend](),
    )
    return r, [
        ("", lambda: r.plot(show=False)),
        ("_group_time", lambda: r.plot_group_time(show=False)),
    ]


def build_panel(backend):
    rng = np.random.default_rng(42)
    rows = []
    for u_idx in range(10):
        unit_effect = rng.normal()
        for t in range(20):
            treatment = 1 if (t >= 10 and u_idx < 5) else 0
            x1 = rng.normal()
            y = unit_effect + 0.1 * t + 2.0 * treatment + 0.5 * x1 + 0.1 * rng.normal()
            rows.append(
                {
                    "unit": f"u{u_idx}",
                    "time": t,
                    "treatment": treatment,
                    "x1": x1,
                    "y": y,
                }
            )
    df = pd.DataFrame(rows)
    r = cp.PanelRegression(
        data=df,
        formula="y ~ C(unit) + C(time) + treatment + x1",
        unit_fe_variable="unit",
        time_fe_variable="time",
        fe_method="dummies",
        model={"pymc": lr, "ols": skl_lr}[backend](),
    )
    return r, [("", lambda: r.plot(show=False))]


def build_sdid(backend):
    model = {
        "pymc": lambda: cp.pymc_models.SyntheticDifferenceInDifferencesWeightFitter(
            sample_kwargs=SAMPLE_KWARGS
        ),
        # OLS estimation is NotImplemented in SDiD's algorithm(), so there is
        # no OLS plot to baseline.
    }[backend]()
    r = cp.SyntheticDifferenceInDifferences(
        cp.load_data("sc"),
        70,
        control_units=["a", "b", "c", "d", "e", "f", "g"],
        treated_units=["actual"],
        model=model,
    )
    return r, [("", lambda: r.plot(show=False))]


def build_prepostnegd(backend):
    model = {"pymc": lr}[backend]()  # Bayes-only experiment
    r = cp.PrePostNEGD(
        cp.load_data("anova1"),
        formula="post ~ 1 + C(group) + pre",
        group_variable_name="group",
        pretreatment_variable_name="pre",
        model=model,
    )
    return r, [("", lambda: r.plot(show=False))]


def build_rkink(backend):
    model = {"pymc": lr}[backend]()  # Bayes-only experiment
    kink = 0.5
    rng = np.random.default_rng(42)
    N = 50
    beta = [0, -1, 0, 2, 0]
    sigma = 0.05
    x = rng.uniform(-1, 1, N)
    ep = rng.normal(0, sigma, N)
    treated = x >= kink
    y = (
        beta[0]
        + beta[1] * x
        + beta[2] * x**2
        + beta[3] * (x - kink) * treated
        + beta[4] * (x - kink) ** 2 * treated
        + ep
    )
    df = pd.DataFrame({"x": x, "y": y, "treated": treated})
    r = cp.RegressionKink(
        df,
        formula=f"y ~ 1 + x + I((x-{kink})*treated)",
        model=model,
        kink_point=kink,
    )
    return r, [("", lambda: r.plot(show=False))]


BUILDERS = {
    "its": build_its,
    "sc": build_sc,
    "did": build_did,
    "piecewise_its": build_piecewise_its,
    "rd": build_rd,
    "staggered_did": build_staggered_did,
    "panel": build_panel,
    "sdid": build_sdid,
    "prepostnegd": build_prepostnegd,
    "rkink": build_rkink,
}

BACKENDS = ("pymc", "ols")


def to_figure(obj) -> plt.Figure:
    """Normalise a plot method's return value to a matplotlib Figure."""
    if isinstance(obj, tuple):
        return obj[0]
    if isinstance(obj, plt.Figure):
        return obj
    raise TypeError(f"Cannot turn {type(obj)!r} into a Figure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="baseline",
        help="Suffix for saved images, e.g. 'baseline' or 'after'.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(BUILDERS),
        help="Restrict to these experiment keys (default: all).",
    )
    parser.add_argument(
        "--backend",
        nargs="*",
        choices=BACKENDS,
        help="Restrict to these backends (default: all).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = args.only or list(BUILDERS)
    backends = args.backend or list(BACKENDS)

    saved, skipped, failed = [], [], []
    for key in keys:
        for backend in backends:
            try:
                result, plots = BUILDERS[key](backend)
            except KeyError:
                skipped.append((key, backend))
                print(f"  SKIP {key}_{backend} (backend unsupported)")
                continue
            except Exception as exc:  # noqa: BLE001 - report, don't abort the run
                failed.append((f"{key}_{backend}", exc))
                print(f"  FAIL {key}_{backend}: {exc}")
                traceback.print_exc()
                continue
            for suffix, call in plots:
                name = f"{key}_{backend}{suffix}"
                try:
                    fig = to_figure(call())
                    path = OUT_DIR / f"{name}_{args.tag}.png"
                    fig.savefig(path, dpi=100, bbox_inches="tight")
                    plt.close(fig)
                    saved.append(path)
                    print(f"  OK   {path}")
                except Exception as exc:  # noqa: BLE001
                    failed.append((name, exc))
                    print(f"  FAIL {name}: {exc}")
                    traceback.print_exc()

    print(f"\nSaved {len(saved)} image(s) to {OUT_DIR}/ (tag={args.tag!r})")
    if failed:
        print(f"\n{len(failed)} plot(s) FAILED:")
        for name, exc in failed:
            print(f"  {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
