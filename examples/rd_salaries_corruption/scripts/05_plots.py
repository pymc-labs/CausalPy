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

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("out/results_stage123.json"))
H = json.load(open("out/results_hier.json"))
bench = R["benchmark"]
paper = -0.078

# ---- fig 1: binned RD scatter (the picture) ----
d = pd.read_parquet("rd_unique.parquet").dropna(subset=["cri2", "margin"])
d = d[d.margin.abs() < 0.3]
bins = np.linspace(-0.3, 0.3, 25)
d["bin"] = pd.cut(d.margin, bins)
g = (
    d.groupby("bin", observed=True)
    .agg(x=("margin", "mean"), y=("cri2", "mean"), n=("cri2", "size"))
    .dropna()
)
fig, ax = plt.subplots(figsize=(8, 5))
left, right = g[g.x < 0], g[g.x >= 0]
ax.scatter(
    left.x,
    left.y,
    s=left.n / 3,
    color="#2c7fb8",
    label="below threshold (lower salary)",
)
ax.scatter(
    right.x,
    right.y,
    s=right.n / 3,
    color="#d95f0e",
    label="above threshold (higher salary)",
)
for side, c in [(d[d.margin < 0], "#2c7fb8"), (d[d.margin >= 0], "#d95f0e")]:
    z = np.polyfit(side.margin, side.cri2, 1)
    xs = np.linspace(side.margin.min(), side.margin.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=c, lw=2)
ax.axvline(0, color="k", ls="--", alpha=0.6)
ax.set_xlabel("Population margin to salary threshold (0 = cutoff)")
ax.set_ylabel("Procurement corruption risk (cri2)")
ax.set_title(
    "Regression discontinuity: corruption risk drops where mayors get a salary raise"
)
ax.legend()
fig.savefig("out/fig_binned_rd.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- fig 2: polynomial order instability (global, wide window, no covariates) ----
ff = R["stage1_functional_form"]
orders = ["poly1", "poly2", "poly3"]
means = [ff[o]["tau_mean"] for o in orders]
lo = [ff[o]["hdi3"] for o in orders]
hi = [ff[o]["hdi97"] for o in orders]
fig, ax = plt.subplots(figsize=(7, 4.2))
xpos = range(len(orders))
ax.errorbar(
    xpos,
    means,
    yerr=[np.array(means) - np.array(lo), np.array(hi) - np.array(means)],
    fmt="o",
    capsize=4,
    color="#555",
)
ax.axhline(0, color="k", ls=":")
ax.axhline(paper, color="red", ls="--", label="paper local-linear estimate (-0.078)")
ax.set_xticks(list(xpos))
ax.set_xticklabels(["linear", "quadratic", "cubic"])
ax.set_ylabel("estimated discontinuity in cri2")
ax.set_title(
    "Global polynomials over a wide window are unreliable\n(Gelman & Imbens 2019)"
)
ax.legend(fontsize=8)
fig.savefig("out/fig_poly_orders.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- fig 3: forest of the credible estimates across methods ----
rows = []
rows.append(
    (
        "rdrobust (paper spec, +covariates)",
        bench[0]["tau_bias_corrected"],
        bench[0]["tau_bias_corrected"] - 1.96 * bench[0]["se_robust"],
        bench[0]["tau_bias_corrected"] + 1.96 * bench[0]["se_robust"],
        "freq",
    )
)
for bw in ["0.05", "0.075", "0.1115", "0.2"]:
    k = f"bw_{bw}"
    s = R["stage1b_bandwidth"][k]
    rows.append(
        (
            f"CausalPy local-linear, bw={bw}",
            s["tau_mean"],
            s["hdi3"],
            s["hdi97"],
            "bayes",
        )
    )
hc = H["tau_cri2_scale"]
rows.append(
    (
        "Hierarchical Beta + kernel (covariate-adj.)",
        hc["mean"],
        hc["hdi3"],
        hc["hdi97"],
        "bayes",
    )
)
fig, ax = plt.subplots(figsize=(9, 5))
for i, (lab, m, l, h, kind) in enumerate(rows[::-1]):
    c = "#d95f0e" if kind == "freq" else "#2c7fb8"
    ax.plot([l, h], [i, i], color=c, lw=2)
    ax.scatter([m], [i], color=c, zorder=3)
    ax.text(h + 0.003, i, f"{m:+.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
ax.axvline(0, color="k", ls=":")
ax.axvline(paper, color="red", ls="--", alpha=0.5, label="paper -0.078")
ax.set_xlabel("discontinuity in corruption risk (cri2) at the salary threshold")
ax.set_title(
    "Effect of a mayoral salary raise on procurement corruption risk\n(negative = less corruption)"
)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("out/fig_forest.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("SAVED fig_binned_rd.png, fig_poly_orders.png, fig_forest.png")
