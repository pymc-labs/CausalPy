# Method Capability Matrix

Use this as a routing-time gate before handing off to `running-causalpy-experiments`. It is not a replacement for method-specific execution guidance.

| Experiment | Data topology | Backend support | Standard outputs | Routing caution |
|---|---|---|---|---|
| `InterruptedTimeSeries` | Single outcome time series | Bayesian and sklearn/OLS | `summary()`, `effect_summary()`, `plot()` | Pre-period forecast counterfactual; use `PiecewiseITS` for explicit full-series level/slope changes. |
| `PiecewiseITS` | Single outcome time series with `step()` or `ramp()` terms | Bayesian and sklearn/OLS | `summary()`, `effect_summary()`, `plot()` | Known interruptions only; not changepoint discovery. |
| `DifferenceInDifferences` | Long treated/control pre/post data with common intervention timing | Bayesian and sklearn/OLS | `summary()`, `effect_summary()`, `plot()` | Not for staggered adoption. |
| `StaggeredDifferenceInDifferences` | Long unit-time panel with different treatment times | Bayesian and sklearn/OLS | `summary()`, `effect_summary()`, `plot()` | Treatment must be absorbing; each calendar period needs at least one untreated unit. |
| `SyntheticControl` | Wide unit-by-time panel with treated and donor units | Bayesian and sklearn/OLS | `summary()`, `effect_summary()`, `plot()` | Requires donor support and credible pre-period fit. |
| `SyntheticDifferenceInDifferences` | Wide unit-by-time panel with treated and donor units | Bayesian in practice | `summary()`, `effect_summary()`, `plot()` | Use only when unit weights and pre-period time weights are part of the design. |
| `PanelRegression` | Long panel with unit and optional time identifiers | Bayesian and sklearn/OLS, but no default model | `summary()`, `print_coefficients()`, `plot()` | `effect_summary()` is not implemented; not a causal design by itself. |
| `PrePostNEGD` | Pretest/posttest nonequivalent group data | Bayesian only | `summary()`, `effect_summary()`, `plot()` | For one pre and one post outcome, not repeated panel trends. |
| `RegressionDiscontinuity` | Running-variable data with sharp cutoff | Bayesian and sklearn/OLS | `summary()`, `effect_summary()`, `plot()` | Sharp RD only; fuzzy RD is not directly implemented. |
| `RegressionKink` | Running-variable data with slope change at kink | Bayesian only | `summary()`, `effect_summary()`, `plot()` | Slope change, not level jump. |
| `InstrumentalVariable` | Outcome/treatment data plus instrument data | Bayesian only | No unified `summary()`, `effect_summary()`, or `plot()` | Requires credible instrument; reporting needs custom inspection. |
| `InversePropensityWeighting` | Observational binary treatment with confounders | Bayesian only | `plot_ate()`, `plot_balance_ecdf()` | No unified `plot()` and no `effect_summary()`; requires overlap. |

## Backend Gate

If the user explicitly requires sklearn/OLS, prefer only methods marked Bayesian and sklearn/OLS. Do not route to `PrePostNEGD`, `RegressionKink`, `InstrumentalVariable`, or `InversePropensityWeighting`. Treat `SyntheticDifferenceInDifferences` as Bayesian-only unless the execution docs say otherwise.

## Reporting Gate

If the user explicitly needs a decision-ready `effect_summary()`, do not route to `PanelRegression`, `InstrumentalVariable`, or `InversePropensityWeighting` without warning. If they need a unified `plot()`, do not route to `InstrumentalVariable` or `InversePropensityWeighting` without warning.
