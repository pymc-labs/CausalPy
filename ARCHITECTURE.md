# ARCHITECTURE

CausalPy implements 10+ quasi-experimental causal inference methods over two core statistical backends (PyMC and scikit-learn), plus an optional third backend (`pymc-forecast`, currently `InterruptedTimeSeries` only). This document orients agents and contributors to where things live, how the pieces compose, and the non-obvious conventions.

## Module Map

| Path | Purpose |
|------|---------|
| `causalpy/__init__.py` | Public API — re-exports experiment classes, models, pipeline, steps, transforms |
| `causalpy/experiments/` | One experiment class per file; `base.py` holds `BaseExperiment` |
| `causalpy/pymc_models.py` | All `PyMCModel` subclasses (Bayesian backend) |
| `causalpy/skl_models.py` | `ScikitLearnAdaptor` mixin, `create_causalpy_compatible_class()` |
| `causalpy/pymc_forecast_models.py` | `PyMCForecastModel` wrapper (optional `pymc-forecast` backend) |
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

## Public API Policy

CausalPy assigns every supported surface to one of four tiers. Membership comes from the manifests below, not from importability or a name's underscore convention. When the same object has more than one documented path, the strongest applicable tier governs its compatibility promise.

1. **Tier 1 — stable top-level API.** `causalpy.__all__` is the authoritative manifest: only its names are stable as `causalpy.<name>`. `docs/source/api/index.md` documents that package with a Sphinx `automodule` using `:members:`, `:undoc-members:`, and `:imported-members:`, and `scripts/check_public_exports.py` statically requires both that documentation wiring and a local binding for every export. Removing, renaming, or incompatibly changing a Tier 1 signature, return contract, or behavior is SemVer-major work; issue a deprecation warning in a prior feature release whenever practical and provide migration or release-note guidance. Adding a Tier 1 name is backwards-compatible but creates this promise.

2. **Tier 2 — documented submodule API.** A qualified symbol is Tier 2 only when its containing CausalPy module is explicitly listed in `docs/source/api/index.md` and the symbol is rendered on that module's generated Sphinx autosummary page. A whole-submodule re-export, direct import, or unprefixed name alone does not promote a symbol. Tier 2 paths are stable within a release line and may break only in a declared major release; normally warn through a deprecation first, and when that is not practical provide explicit migration and release-note guidance in that major release.

3. **Tier 3 — internal implementation.** Everything not classified as Tier 1, Tier 2, or Tier 4 is internal, including unprefixed support helpers in modules such as `utils`, `plot_utils`, `date_utils`, and `custom_exceptions`. Evaluate the explicit Tier 4 protocol before this fallback, so an integration hook does not become Tier 3 merely because it has an underscore-style name. Leading underscores remain a strong signal, but are not the classifier. Tier 3 paths have no compatibility guarantee; leave ambiguous helpers here until a separately reviewed manifest and documentation decision promotes them, and use a deprecation plan before removing an import path with known users.

4. **Tier 4 — protocol and integrator hooks.** `BaseExperiment.set_maketables_options()`, the `BaseExperiment.__maketables_*__` protocol, and model `_clone()` overrides consumed by CausalPy checks are supported for integrators despite their underscore-style names. They are documented for contributors and integrators here rather than as end-user Sphinx API. An incompatible protocol change is SemVer-major and needs a migration or deprecation notice. No other underscored name is Tier 4 unless this policy is updated.

### Current boundary decisions

`EffectSummary` is the only new Tier 1 promotion: public `effect_summary()` methods return it, and the reporting knowledgebase already names its `table` and `text` contract. Re-exporting it as `causalpy.EffectSummary` gives users a stable type path; adding `reporting` to the Sphinx API surface and its explicit `__all__` documents that type without exposing its underscored calculation helpers.

The top-level Sphinx `automodule` documents every existing `causalpy.__all__` export, including established utility and transform re-exports, without adding their support modules to the autosummary list. This aligns existing Tier 1 paths with docs rather than broadly promoting `utils`, `plot_utils`, `date_utils`, `custom_exceptions`, or other ambiguous helpers. No aliases, import removals, plot signatures, or effect-summary behavior change as part of this policy.

## Backend Model

Backend dispatch is centralized in `causalpy/experiments/model_adapter.py`. `BaseExperiment.__init__` calls `make_model_adapter()`, which handles sklearn coercion (`clone`/`deepcopy`, `create_causalpy_compatible_class()`, `fit_intercept=False` warning), default-model instantiation, and `supports_bayes`/`supports_ols`/`supports_pymc_forecast` validation. Each experiment stores `self._model_backend` (private) and keeps `self.model` as the public handle. Bayesian-only consumers check `ModelAdapter.supports_idata` and use `require_idata()`; the honest `idata` property returns `None` for unsupported or unfitted backends instead of using `AttributeError` as capability discovery.

Standard regression experiments call `self._model_backend.fit(X, y, coords=build_coords(...))` unconditionally. `build_coords()` assembles the PyMC `coeffs` / `obs_ind` / `treated_units` dict; sklearn backends ignore `coords`. `SklearnModelAdapter` normalizes inputs before delegating to sklearn: xarray `DataArray` values become numpy arrays, and a single-column `treated_units` outcome is squeezed to 1D so call sites do not need per-experiment `.isel(treated_units=0)` branches. For estimand calculations, every adapter's `predict_mu()` returns response-scale expected outcomes as an xarray `DataArray` with canonical `("chain", "draw", "obs_ind", "treated_units")` dimensions; sklearn point predictions use singleton chain and draw dimensions. Use `predict()` only when a caller needs the backend's full prediction container, including posterior predictive `y_hat`. Every adapter's `score()` returns a `pd.Series` with one `unit_{i}_r2` entry per treated unit and optional `unit_{i}_r2_std` entries when the backend carries posterior dispersion.

```python
from causalpy.experiments.model_adapter import build_coords

self._model_backend.fit(
    X=X,
    y=y,
    coords=build_coords(self.labels, X.shape[0]),
)
```

Experiments with non-standard fit signatures bypass this path and call `self.model.fit(...)` directly with custom arguments: `InstrumentalVariable` (two-stage IV), `InversePropensityWeighting` (propensity `fit(X, t, coords)`), and `SyntheticDifferenceInDifferences` (dict-shaped weight-fitter inputs). Those models are not forced through `build_coords` or sklearn y-normalization.

`PyMCModel` extends `pymc.Model` with a sklearn-like `fit` / `predict` / `score` interface; the adapter's `predict()` returns response-scale expected outcomes as an `xr.DataArray` with canonical dims `(chain, draw, obs_ind, treated_units)` on every backend (sklearn backends return singleton `chain`/`draw`), and experiments compute impact as plain xarray subtraction `y - predict(X)`. `ScikitLearnAdaptor` is a mixin patched onto any `RegressorMixin` via `create_causalpy_compatible_class()` during adapter construction. Every experiment declares `supports_ols` and `supports_bayes`; validation runs in `make_model_adapter()`. When `model=None`, `_default_model_class` is instantiated (always Bayesian; `PanelRegression` requires an explicit model).

The optional third backend, `PyMCForecastModel` (`causalpy/pymc_forecast_models.py`), wraps a `pymc_forecast` forecasting model behind the same protocol and is wired through `PyMCForecastAdapter`. It reports as Bayesian (`is_bayesian` is true for both `"pymc"` and `"pymc-forecast"` adapter kinds), so experiments and checks that branch on Bayesian-vs-OLS treat it like a PyMC backend. Experiments opt in via `supports_pymc_forecast` (currently `InterruptedTimeSeries` only); the dependency ships as the `causalpy[forecast]` extra, pinned to one upstream minor while `pymc-forecast` is 0.x.

## Experiment Lifecycle

Instantiation fits eagerly in `__init__`: `_build_design_matrices()` → `_prepare_data()` → `algorithm()`. There is no separate `.fit()` on the experiment. Each subclass's public `plot(*, ...)` delegates to `_render_plot()`, which calls the subclass's backend-agnostic `_plot()`. Uncertainty rendering keys on data properties of the canonical prediction container (`has_posterior_draws()`), not on backend identity. `effect_summary()` returns `EffectSummary(table, text)` using helpers from `causalpy.reporting`.

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
| **Formulas** | Patsy `dmatrices()` for design matrices; `build_design_matrices()` for counterfactual prediction. Bare datetime predictors are encoded as continuous elapsed days from the fitted origin; use `C(date)` for date fixed effects. `PiecewiseITS` uses `step()`/`ramp()` stateful transforms. |
| **obs_ind** | All experiments set `data.index.name = "obs_ind"`. Canonical xarray/PyMC dimension name. |
| **treated_units always 2D** | Even single-unit experiments use `treated_units=["unit_0"]`. Never pass 1D y to PyMC. |
| **Impact uses mu, not y_hat** | The adapter's `predict()` extracts posterior `mu` (conditional expected outcome in observed units), not `y_hat` (with observation noise); impact is `y - predict(X)`. For GLMs, `mu` must be inverse-linked before impact; see `docs/source/knowledgebase/prediction-contract.md`. |
| **Intercept handling** | Patsy includes intercept by default. sklearn models must use `fit_intercept=False`. |
| **Eager fitting** | MCMC runs during `__init__`. No lazy `.fit()` on the experiment. |
| **HDI_PROB** | Project default is 0.94 (ArviZ default), not 0.95. |
| **create_causalpy_compatible_class** | Applied during `make_model_adapter()` for sklearn backends; clones the user instance before patching. |

## Adding New Code

Copy the closest existing experiment or model and follow the `BaseExperiment` contract:

- Declare `supports_ols` / `supports_bayes` (and `supports_pymc_forecast` to opt into the optional pymc-forecast backend); implement a single backend-agnostic `_plot()` (and an explicit `get_plot_data(*, ...)` only where that view is supported) that consumes the canonical prediction container, keying uncertainty rendering on `has_posterior_draws()` rather than backend identity
- `algorithm()` with the fit/predict/impact flow; every concrete experiment declares its own explicit `effect_summary(...)` contract, using helpers in `causalpy.reporting` where that summary is implemented
- Public APIs expose explicit named parameters rather than bare `*args` / `**kwargs`; use keyword-only optional controls for public plotting and plot-data APIs (enforced by `causalpy/tests/test_public_signatures.py` and surveyed by `scripts/audit_public_signatures.py`). A genuine dynamic or third-party forwarder requires an `Other Parameters` contract and a narrow structural-test exemption. For experiments without a unified plot view (e.g. `InversePropensityWeighting`, `InstrumentalVariable`), declare an explicit `plot()` stub that raises `NotImplementedError`. For `hdi_prob` defaults, use ``Defaults to :data:`~causalpy.constants.HDI_PROB` (currently 0.94).`` in the docstring.
- Raise `FormulaException`, `DataException`, or `BadIndexException` from `causalpy.custom_exceptions` for formula, data, and index errors
- Avoid backwards-compat shims for APIs introduced in the same PR

**Keeping it current:** When you add, remove, or structurally change an experiment class, PyMC model, backend dispatch path, data contract, or Tier 1 export, update this file and its documented surface in the same PR. `scripts/check_public_exports.py` enforces experiment/check export wiring plus the Tier 1 top-level Sphinx directive, while `scripts/check_architecture_inventory.py` enforces the experiment inventory table (both run via prek); run `make check-exports` / `make check-architecture` locally if needed.
