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

"""Bayesian RD with CausalPy: functional-form study (polynomial orders + spline),
local-linear bandwidth sweep, placebo cutoffs and donut RD.
Reads rd_unique.parquet + benchmark_results.json; writes out/results_stage123.json
and out/fig_rd_causalpy.png."""

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt

import causalpy as cp

os.makedirs("out", exist_ok=True)

RNG = 42


def SK(chains=4, draws=800, tune=800):
    return dict(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=4,
        target_accept=0.9,
        progressbar=False,
        random_seed=RNG,
    )


def model(**kw):
    return cp.pymc_models.LinearRegression(sample_kwargs=SK(**kw))


full = pd.read_parquet("rd_unique.parquet").dropna(subset=["cri2", "margin"]).copy()
full = full.rename(columns={"cri2": "y", "margin": "x"})
full["treated"] = (full.x >= 0).astype(int)
print("unique sample n=", len(full), flush=True)
results = {"benchmark": json.load(open("benchmark_results.json"))}


def dsum(res):
    p = np.asarray(res.discontinuity_at_threshold).flatten()
    return dict(
        tau_mean=round(float(p.mean()), 4),
        tau_sd=round(float(p.std()), 4),
        hdi3=round(float(np.percentile(p, 3)), 4),
        hdi97=round(float(np.percentile(p, 97)), 4),
        p_negative=round(float((p < 0).mean()), 4),
    ), p


# STAGE 1: functional-form study on a manageable subsample, tight window |x|<0.25
# NOTE: CausalPy requires a pre-computed 0/1 `treated` column; it is NOT auto-created.
w = full[full.x.abs() < 0.25].copy()
w = w.sample(n=min(6000, len(w)), random_state=RNG)
print("stage1 functional-form subsample n=", len(w), flush=True)
specs = {
    "poly1": "y ~ 1 + x + treated + x:treated",
    "poly2": "y ~ 1 + x + I(x**2) + treated + x:treated + I(x**2):treated",
    "poly3": "y ~ 1 + x + I(x**2) + I(x**3) + treated + x:treated + I(x**2):treated + I(x**3):treated",
    "spline_bs5": "y ~ 1 + bs(x, df=5) + treated + bs(x, df=5):treated",
}
results["stage1_functional_form"] = {}
for name, fml in specs.items():
    res = cp.RegressionDiscontinuity(
        w[["x", "y", "treated"]],
        formula=fml,
        treatment_threshold=0.0,
        running_variable_name="x",
        model=model(chains=2),
    )
    s, post = dsum(res)
    results["stage1_functional_form"][name] = s
    np.save(f"out/post_{name}.npy", post)
    print(name, s, flush=True)
    json.dump(results, open("out/results_stage123.json", "w"), indent=2, default=str)

# STAGE 1b: local-linear bandwidth sweep on FULL data (the credible estimates)
results["stage1b_bandwidth"] = {}
for bw in [0.05, 0.075, 0.1115, 0.2]:
    res = cp.RegressionDiscontinuity(
        full[["x", "y", "treated"]],
        formula="y ~ 1 + x + treated + x:treated",
        treatment_threshold=0.0,
        running_variable_name="x",
        bandwidth=bw,
        model=model(),
    )
    s, post = dsum(res)
    s["n_in_bw"] = int((full.x.abs() <= bw).sum())
    results["stage1b_bandwidth"][f"bw_{bw}"] = s
    np.save(f"out/post_bw_{bw}.npy", post)
    print("bw", bw, s, flush=True)
    json.dump(results, open("out/results_stage123.json", "w"), indent=2, default=str)

# main RD plot (optimal bandwidth)
res_main = cp.RegressionDiscontinuity(
    full[["x", "y", "treated"]],
    formula="y ~ 1 + x + treated + x:treated",
    treatment_threshold=0.0,
    running_variable_name="x",
    bandwidth=0.1115,
    model=model(),
)
try:
    fig, ax = res_main.plot()
    fig.savefig("out/fig_rd_causalpy.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    es = res_main.effect_summary()
    open("out/effect_summary.txt", "w").write(getattr(es, "text", str(es)))
except Exception as e:
    print("plot/effect_summary failed:", e, flush=True)

# STAGE 2: placebo cutoffs (one side only) + donut
results["stage2_validity"] = {}
for label, sub, fake in [
    ("placebo_above_0.3", full[(full.x > 0) & (full.x < 0.6)], 0.3),
    ("placebo_below_-0.3", full[(full.x < 0) & (full.x > -0.6)], -0.3),
]:
    s2 = sub.sample(n=min(6000, len(sub)), random_state=RNG).copy()
    s2["treated"] = (s2.x >= fake).astype(int)
    if s2.treated.nunique() < 2:
        results["stage2_validity"][label] = {"skipped": 1}
        continue
    res = cp.RegressionDiscontinuity(
        s2[["x", "y", "treated"]],
        formula="y ~ 1 + x + treated + x:treated",
        treatment_threshold=fake,
        running_variable_name="x",
        model=model(chains=2),
    )
    s, _ = dsum(res)
    results["stage2_validity"][label] = s
    print(label, s, flush=True)
res = cp.RegressionDiscontinuity(
    full[["x", "y", "treated"]],
    formula="y ~ 1 + x + treated + x:treated",
    treatment_threshold=0.0,
    running_variable_name="x",
    bandwidth=0.1115,
    donut_hole=0.02,
    model=model(),
)
s, _ = dsum(res)
results["stage2_validity"]["donut_0.02"] = s
print("donut", s, flush=True)
json.dump(results, open("out/results_stage123.json", "w"), indent=2, default=str)
print("SAVED out/results_stage123.json", flush=True)
