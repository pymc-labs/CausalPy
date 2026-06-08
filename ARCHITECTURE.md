# ARCHITECTURE

CausalPy implements 10+ quasi-experimental causal inference methods over two statistical backends (PyMC and scikit-learn). This document orients agents and contributors to where things live, how the pieces compose, and the non-obvious conventions.

## Module Map

| Path | Purpose |
|------|---------|
| `causalpy/__init__.py` | Public API — re-exports experiment classes, models, pipeline, steps, transforms |
| `causalpy/experiments/` | One experiment class per file; `base.py` holds `BaseExperiment` |
| `causalpy/pymc_models.py` | All `PyMCModel` subclasses (Bayesian backend) |
| `causalpy/skl_models.py` | `ScikitLearnAdaptor` mixin, `create_causalpy_compatible_class()` |
| `causalpy/reporting.py` | `EffectSummary`, statistics and prose for both backends |
| `causalpy/pipeline.py` | `Pipeline`, `PipelineContext`, `PipelineResult`, `Step` protocol |
| `causalpy/steps/` | `EstimateEffect`, `SensitivityAnalysis`, `GenerateReport` |
| `causalpy/checks/` | Diagnostic checks (`PlaceboInTime`, `LeaveOneOut`, `ConvexHullCheck`, etc.) |
| `causalpy/transforms.py` | Patsy `step()` / `ramp()` transforms for piecewise ITS |
| `causalpy/data/` | `load_data()` and synthetic dataset generators |
| `causalpy/tests/` | pytest suite |
| Support modules | `constants.py`, `custom_exceptions.py`, `utils.py`, `plot_utils.py`, `date_utils.py`, `variable_selection_priors.py`, `maketables_adapters.py` |
| `docs/source/notebooks/` | How-to notebooks (`{method}_{backend}.ipynb`) |
| `docs/source/knowledgebase/` | Educational content (glossary, reporting explainers) |

## Two-Backend Model

Dispatch is via `isinstance` checks throughout `BaseExperiment` and experiment subclasses:

```python
if isinstance(self.model, PyMCModel):
    # Bayesian path — xarray DataArrays, InferenceData
elif isinstance(self.model, RegressorMixin):
    # OLS path — numpy arrays
```

`PyMCModel` extends `pymc.Model` with a sklearn-like `fit` / `predict` / `score` / `calculate_impact` interface. `ScikitLearnAdaptor` is a mixin patched onto any `RegressorMixin` via `create_causalpy_compatible_class()` (mutates the instance in place).

Every experiment declares `supports_ols` and `supports_bayes`; `BaseExperiment.__init__` validates the model type. When `model=None`, `_default_model_class` is instantiated (always Bayesian; `PanelRegression` requires an explicit model).

## Experiment Lifecycle

Instantiation fits eagerly in `__init__`: `_build_design_matrices()` → `_prepare_data()` → `algorithm()`. There is no separate `.fit()` on the experiment. Each subclass's public `plot(*, ...)` delegates to `_render_plot()`, which dispatches to `_bayesian_plot()` or `_ols_plot()`. `effect_summary()` returns `EffectSummary(table, text)` using helpers from `causalpy.reporting`.

## Experiment Inventory

| Class | Method | Backends | Notable quirk |
|-------|--------|----------|---------------|
| `InterruptedTimeSeries` | ITS | OLS + Bayes | 3-period design via `treatment_end_time` |
| `PiecewiseITS` | Segmented ITS | OLS + Bayes | Fits full series; `step()`/`ramp()` transforms |
| `DifferenceInDifferences` | DiD | OLS + Bayes | Effect from interaction coefficient |
| `StaggeredDifferenceInDifferences` | Staggered DiD | OLS + Bayes | Fits untreated obs only |
| `SyntheticControl` | SC | OLS + Bayes | Multi-unit; control/treated unit lists, no formula |
| `SyntheticDifferenceInDifferences` | SDiD | OLS + Bayes | Tau computed analytically from weight posteriors |
| `RegressionDiscontinuity` | RD | OLS + Bayes | `epsilon` at threshold; optional `bandwidth` |
| `RegressionKink` | RKD | Bayes only | Slope change at `kink_point` |
| `PrePostNEGD` | Pretest/posttest | Bayes only | `group_variable_name` + `pretreatment_variable_name` |
| `InversePropensityWeighting` | IPW | Bayes only | Two-stage; no unified `plot()` |
| `InstrumentalVariable` | IV/2SLS | Bayes only | Non-standard `fit()` signature; no unified `plot()` |
| `PanelRegression` | Panel FE | OLS + Bayes | No `_default_model_class`; model required |

## PyMC Models

- `LinearRegression` — ITS, DiD, RD, RKD, PrePostNEGD, PiecewiseITS, StaggeredDiD, PanelRegression
- `WeightedSumFitter` / `SoftmaxWeightedSumFitter` — SyntheticControl
- `SyntheticDifferenceInDifferencesWeightFitter` — SyntheticDifferenceInDifferences
- `InstrumentalVariableRegression` — InstrumentalVariable
- `PropensityScore` — InversePropensityWeighting
- `BayesianBasisExpansionTimeSeries` / `StateSpaceTimeSeries` — ITS alternatives (experimental)

## Key Conventions

| Topic | Detail |
|-------|--------|
| **Formulas** | Patsy `dmatrices()` for design matrices; `build_design_matrices()` for counterfactual prediction. `PiecewiseITS` uses `step()`/`ramp()` stateful transforms. |
| **obs_ind** | All experiments set `data.index.name = "obs_ind"`. Canonical xarray/PyMC dimension name. |
| **treated_units always 2D** | Even single-unit experiments use `treated_units=["unit_0"]`. Never pass 1D y to PyMC. |
| **Impact uses mu, not y_hat** | `calculate_impact()` subtracts posterior `mu` (expected value), not `y_hat` (with observation noise). |
| **Intercept handling** | Patsy includes intercept by default. sklearn models must use `fit_intercept=False`. |
| **Eager fitting** | MCMC runs during `__init__`. No lazy `.fit()` on the experiment. |
| **HDI_PROB** | Project default is 0.94 (ArviZ default), not 0.95. |
| **create_causalpy_compatible_class** | Mutates the passed sklearn instance in place; does not return a new object. |

## Adding New Code

Copy the closest existing experiment or model and follow the `BaseExperiment` contract:

- Declare `supports_ols` / `supports_bayes`; implement `_bayesian_plot()` / `_ols_plot()` only for supported backends
- `algorithm()` with the fit/predict/impact flow; `effect_summary()` via helpers in `causalpy.reporting`
- Public `plot(*, ...)` with a kwarg-only signature that delegates to `_render_plot()` — bare `*args` / `**kwargs` are forbidden on the public surface (enforced by `causalpy/tests/test_public_plot_signatures.py`). For experiments without a unified plot view (e.g. `InversePropensityWeighting`, `InstrumentalVariable`), declare an explicit `plot()` stub that raises `NotImplementedError`. For `hdi_prob` defaults, use ``Defaults to :data:`~causalpy.constants.HDI_PROB` (currently 0.94).`` in the docstring.
- Raise `FormulaException`, `DataException`, or `BadIndexException` from `causalpy.custom_exceptions` for formula, data, and index errors
- Avoid backwards-compat shims for APIs introduced in the same PR

**Keeping it current:** When you add, remove, or structurally change an experiment class, PyMC model, backend dispatch path, or data contract, update this file in the same PR.
