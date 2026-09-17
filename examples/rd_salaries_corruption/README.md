# RD replication — politicians' salaries and corruption

A Bayesian regression-discontinuity replication, built with CausalPy, of:

> Klašnja, M., Fazekas, M., & Alshaibani, A. (2026). *Revisiting the Link between Politicians' Salaries and
> Corruption.* British Journal of Political Science.

Mayoral salaries in 11 EU countries jump discretely at population thresholds; the design asks whether the
salary raise reduces procurement corruption risk (`cri2 ∈ [0,1]`). Running variable `margin` = % distance to the
nearest threshold (cutoff 0); treatment = above threshold.

## Contents

| Path | What |
|---|---|
| `rd_pymc_salaries_corruption.ipynb` | Runnable notebook: CausalPy RD, bandwidth sweep, polynomial-instability demo, a hierarchical kernel-weighted Beta-likelihood RD, and validity checks. |
| `REPORT.md` | Written findings and honest evaluation (the headline result, robustness, and where it is fragile). |
| `figures/` | Pre-rendered figures used by `REPORT.md`. |
| `data/rd_salaries_corruption.csv.gz` | Compact extract (unique-threshold sample, 38,663 contract rows) so the notebook runs without the 896 MB raw file. Columns: `y` (=cri2), `x` (=margin), `citycode`, `country`, `year`, `treated`. |
| `scripts/` | Full pipeline from the raw Dataverse data (see below). |

## Reproducing from the raw data

The bundled CSV is enough to run the notebook. To regenerate everything from source:

1. Download + unzip the archive from Harvard Dataverse
   [doi:10.7910/DVN/TESJMM](https://doi.org/10.7910/DVN/TESJMM) (`data-main.zip` → `data-main.dta`).
2. `pip install causalpy rdrobust pyreadstat pyarrow`
3. Run, in order:
   - `scripts/01_prep.py` — build `rd_unique.parquet` + `analysis_full2.parquet` from `data-main.dta`.
   - `scripts/02_benchmark_rdrobust.py` — reproduce the authors' Table 2 `rdrobust` estimates → `benchmark_results.json`.
   - `scripts/03_causalpy_rd.py` — CausalPy RD: functional-form study, bandwidth sweep, placebo + donut.
   - `scripts/04_hierarchical_beta.py` — the hierarchical, kernel-weighted, Beta-likelihood RD model.
   - `scripts/05_plots.py` — render the figures.

## Headline

The paper's direction replicates robustly — a mayoral salary raise is associated with **lower** procurement
corruption risk (negative at posterior probability ≈ 1.0 across bandwidths and in the covariate-adjusted
hierarchical Beta model; frequentist `rdrobust` reproduces τ = −0.078 exactly). The **precise magnitude is
specification-dependent and fragile** (mass points / heaping in the pooled-threshold running variable); see
`REPORT.md` §6–7. Independent replication, not peer-reviewed.
