# ARCHITECTURE

Internal architecture reference for AI agents and contributors working on CausalPy.

## Module Map

| Path | Purpose |
|------|---------|
| `causalpy/__init__.py` | Public API surface — re-exports all experiment classes, models, pipeline, steps, transforms |
| `causalpy/experiments/` | Package of experiment classes (one per file) plus `base.py` |
| `causalpy/experiments/base.py` | `BaseExperiment` ABC — dispatch logic, `_render_plot`, maketables hooks |
| `causalpy/pymc_models.py` | All `PyMCModel` subclasses (Bayesian backend) |
| `causalpy/skl_models.py` | `ScikitLearnAdaptor` mixin, `WeightedProportion`, `create_causalpy_compatible_class()` |
| `causalpy/reporting.py` | `EffectSummary` dataclass, statistics computation, prose generation for both backends |
| `causalpy/maketables_adapters.py` | Backend-specific adapters for optional `maketables` table export |
| `causalpy/pipeline.py` | `Pipeline`, `PipelineContext`, `PipelineResult`, `Step` protocol |
| `causalpy/steps/` | Pipeline steps: `EstimateEffect`, `SensitivityAnalysis`, `GenerateReport` |
| `causalpy/checks/` | Diagnostic checks: `PlaceboInTime`, `PlaceboInSpace`, `LeaveOneOut`, `ConvexHullCheck`, `BandwidthSensitivity`, `McCraryDensityTest`, `PriorSensitivity`, `PersistenceCheck`, `PreTreatmentPlaceboCheck` |
| `causalpy/transforms.py` | Patsy stateful transforms `step()` and `ramp()` for piecewise ITS |
| `causalpy/variable_selection_priors.py` | Spike-and-slab and horseshoe priors for IV variable selection |
| `causalpy/constants.py` | `HDI_PROB` (0.94), `LEGEND_FONT_SIZE` (12) |
| `causalpy/custom_exceptions.py` | `BadIndexException`, `FormulaException`, `DataException` |
| `causalpy/utils.py` | Shared helpers: `round_num`, `_as_scalar`, `extract_lift_for_mmm`, `plot_correlations`, formula parsing utils |
| `causalpy/plot_utils.py` | `plot_xY` (HDI ribbon helper), `get_hdi_to_df` |
| `causalpy/date_utils.py` | Date axis formatting for matplotlib (`format_date_axes`, `_combine_datetime_indices`) |
| `causalpy/data/` | `load_data()` and `simulate_data` module for example/synthetic datasets |
| `causalpy/version.py` | `__version__` string |
| `causalpy/tests/` | pytest suite — integration tests per backend, unit tests for models/reporting/checks |
| `docs/source/notebooks/` | Jupyter how-to notebooks (named `{method}_{backend}.ipynb`) |
| `docs/source/knowledgebase/` | Educational content (glossary, reporting statistics explainers) |

## Backend Architecture

CausalPy supports two model backends dispatched via `isinstance` checks:

### Dispatch Pattern

```python
if isinstance(self.model, PyMCModel):
    # Bayesian path — xarray DataArrays, InferenceData
elif isinstance(self.model, RegressorMixin):
    # OLS path — numpy arrays
```

This pattern appears in `BaseExperiment.__init__` (validation), `_render_plot` (plotting), `get_plot_data` (data export), and each experiment's `algorithm()` and `effect_summary()`.

### PyMCModel (Bayesian backend)

`PyMCModel` extends `pymc.Model` and provides a sklearn-like interface:

| Method | Contract |
|--------|----------|
| `build_model(X, y, coords)` | Define PyMC model graph. Must register `pm.Data("X", ...)` and `pm.Data("y", ...)`, create `mu` Deterministic and `y_hat` likelihood with dims `["obs_ind", "treated_units"]` |
| `fit(X, y, coords)` | Calls `build_model`, then `pm.sample`, `sample_prior_predictive`, `sample_posterior_predictive`. Returns `az.InferenceData` |
| `predict(X)` | Calls `_data_setter(X)` then `sample_posterior_predictive` for `["y_hat", "mu"]`. Returns `az.InferenceData` |
| `score(X, y)` | Computes Bayesian R² per treated unit. Returns `pd.Series` |
| `calculate_impact(y_true, y_pred)` | `y_true - y_pred["posterior_predictive"]["mu"]` (uses mu, NOT y_hat) |
| `print_coefficients(labels)` | Prints posterior mean + HDI for each coefficient |

Priors use `pymc_extras.Prior` objects. Priority: user-specified > `priors_from_data()` > `default_priors` class attribute.

### ScikitLearnAdaptor (OLS backend)

`ScikitLearnAdaptor` is a mixin class providing CausalPy-compatible methods:

- `calculate_impact(y_true, y_pred)` → `y_true - y_pred` (numpy subtraction)
- `calculate_cumulative_impact(impact)` → `np.cumsum(impact)`
- `print_coefficients(labels)` → prints `coef_` values
- `get_coeffs()` → `np.squeeze(self.coef_)`

`create_causalpy_compatible_class(estimator)` takes an instantiated sklearn `RegressorMixin` and monkey-patches `ScikitLearnAdaptor` methods onto it via `_add_mixin_methods`. Returns the mutated instance (not a new class).

### supports_ols / supports_bayes

Every experiment class declares these as class attributes. `BaseExperiment.__init__` raises `ValueError` if the wrong model type is passed.

### _default_model_class

When `model=None` is passed to an experiment, `BaseExperiment.__init__` instantiates `_default_model_class()` with no arguments. This always produces a Bayesian model. Experiments without a default (e.g. `PanelRegression`) raise `ValueError` if `model=None`.

## Experiment Lifecycle

### 1. Instantiation

```text
ExperimentClass(data, ..., model=None)
  → BaseExperiment.__init__(model)
      → wrap sklearn model via create_causalpy_compatible_class() if needed
      → instantiate _default_model_class if model is None
      → validate supports_ols / supports_bayes
  → self.data = data; self.formula = formula
  → input_validation(...)
  → _build_design_matrices()
  → _prepare_data()  (convert to xarray DataArrays)
  → algorithm()
```

Most experiments fit eagerly in `__init__` — instantiation triggers the full MCMC run. There is no separate `.fit()` on the experiment (only on the model).

### 2. _build_design_matrices()

Uses patsy `dmatrices(formula, data_pre)` for pre-intervention data, stores `design_info`. Uses `build_design_matrices([y_design_info, x_design_info], data_post)` for post-intervention data — this ensures consistent encoding for out-of-sample prediction.

### 3. _prepare_data()

Converts numpy design matrices into `xr.DataArray` with dims `["obs_ind", "coeffs"]` for X and `["obs_ind", "treated_units"]` for y.

### 4. algorithm()

Per-experiment fitting logic:
1. `model.fit(X_pre, y_pre, coords)` — train on pre-intervention data
2. `model.score(X_pre, y_pre)` — evaluate fit quality
3. `model.predict(X_pre)` — in-sample predictions
4. `model.predict(X_post)` — counterfactual predictions
5. `model.calculate_impact(y_post, post_pred)` — causal effect
6. `model.calculate_cumulative_impact(impact)` — cumulative effect

### 5. _render_plot()

Template method called by each subclass's public `plot()`:
1. Applies `arviz-darkgrid` style context
2. Dispatches to `_bayesian_plot(**draw_kwargs)` or `_ols_plot(**draw_kwargs)` based on model type
3. Applies `legend_kwargs` in-place to preserve custom handles
4. Optionally calls `plt.show()`

### 6. effect_summary()

Abstract method on `BaseExperiment`. Each subclass implements it using helpers from `causalpy.reporting`:
- Bayesian: `_compute_statistics()` → `_generate_table()` → `_generate_prose_detailed()`
- OLS: `_compute_statistics_ols()` → `_generate_table_ols()` → `_generate_prose_detailed_ols()`

Returns `EffectSummary(table=pd.DataFrame, text=str)`.

### 7. get_plot_data()

Dispatches to `get_plot_data_bayesian()` or `get_plot_data_ols()`. Returns a `pd.DataFrame` with columns for predictions, impacts, and HDI bounds.

## Formula and Data Pipeline

### patsy workflow

1. `dmatrices(formula, df_pre)` → `(y, X)` numpy DesignMatrix objects
2. Store `y.design_info` and `X.design_info` for later reuse
3. `build_design_matrices([y_design_info, x_design_info], df_post)` → counterfactual matrices with consistent factor encoding
4. `self.labels = X.design_info.column_names` — coefficient names

### Custom transforms (PiecewiseITS)

`step(time, threshold)` → binary indicator `(time >= threshold)`
`ramp(time, threshold)` → `max(0, time - threshold)`

Both are patsy `stateful_transform` objects that memorize datetime origin during first pass and convert datetime to numeric days internally.

### obs_ind index naming

All experiments rename `data.index.name = "obs_ind"`. This is the canonical dimension name for xarray DataArrays and PyMC model coordinates.

## Data Contracts

### PyMC Backend

| Object | Type | Dims/Coords |
|--------|------|-------------|
| X (input) | `xr.DataArray` | `["obs_ind", "coeffs"]` with coord values |
| y (input) | `xr.DataArray` | `["obs_ind", "treated_units"]` — always 2D |
| coords dict | `dict` | Keys: `"coeffs"`, `"obs_ind"`, `"treated_units"` (required) |
| fit() return | `az.InferenceData` | Contains posterior, prior_predictive, posterior_predictive |
| predict() return | `az.InferenceData` | `posterior_predictive` group with `mu` and `y_hat` vars |
| mu | `xr.DataArray` | Deterministic mean; dims `["chain", "draw", "obs_ind", "treated_units"]` |
| y_hat | `xr.DataArray` | Observation with noise; same dims as mu |
| impact | `xr.DataArray` | `y_true - mu`; trailing dim is `"obs_ind"` |

Key conventions:
- `treated_units` dim is **always present and 2D** even for single-unit experiments (value: `["unit_0"]`)
- Impact uses `mu` (posterior expectation), NOT `y_hat` (with observation noise)
- `coeffs_raw` dim appears in softmax models (N-1 logits, first pinned to zero)

### sklearn Backend

| Object | Type | Notes |
|--------|------|-------|
| X (input) | `np.ndarray` or `xr.DataArray` | 2D, shape `(n_obs, n_features)` |
| y (input) | `np.ndarray` or 1D `xr.DataArray` | `.isel(treated_units=0)` before passing to sklearn |
| predict() return | `np.ndarray` | Shape `(n_obs, 1)` or `(n_obs,)` |
| coef_ | `np.ndarray` | Accessed via `get_coeffs()` → squeezed |
| impact | `np.ndarray` | Simple `y_true - y_pred` |

## Experiment Inventory

| Class | Causal Method | `supports_ols` | `supports_bayes` | Default Model | Notable Quirks |
|-------|--------------|-----------------|-------------------|---------------|----------------|
| `InterruptedTimeSeries` | ITS (pre/post fit) | Yes | Yes | `LinearRegression` | Supports 3-period design via `treatment_end_time`; eager fit in `__init__` |
| `PiecewiseITS` | ITS (segmented regression) | Yes | Yes | `LinearRegression` | Fits full time series (not pre-only); uses `step()`/`ramp()` transforms |
| `DifferenceInDifferences` | DiD | Yes | Yes | `LinearRegression` | Fits all data (no pre/post split); effect from interaction coefficient |
| `StaggeredDifferenceInDifferences` | Staggered DiD (BJS imputation) | Yes | Yes | `LinearRegression` | Fits untreated observations only; validates absorbing treatment |
| `SyntheticControl` | SC | Yes | Yes | `WeightedSumFitter` | Multi-unit (multiple `treated_units`); no formula — uses control/treated unit lists |
| `SyntheticDifferenceInDifferences` | SDiD | Yes | Yes | `SDiDWeightFitter` | Cut-posterior: tau computed analytically from weight posteriors |
| `RegressionDiscontinuity` | RD (sharp) | Yes | Yes | `LinearRegression` | `epsilon` parameter for causal effect evaluation at threshold; optional `bandwidth` |
| `RegressionKink` | RKD | No | Yes | `LinearRegression` | `kink_point` instead of threshold; evaluates slope change |
| `PrePostNEGD` | Pretest/posttest | No | Yes | `LinearRegression` | Uses `group_variable_name` and `pretreatment_variable_name` |
| `InversePropensityWeighting` | IPW | No | Yes | `PropensityScore` | Non-standard: two-stage (propensity then outcome); no unified `plot()` |
| `InstrumentalVariable` | IV/2SLS | No | Yes | `IVRegression` | Non-standard `fit()` signature (X, Z, y, t, coords, priors); no unified `plot()` |
| `PanelRegression` | Panel FE | Yes | Yes | None (required) | Supports demeaned and dummy-variable FE; no `_default_model_class` |

## PyMC Model Inventory

| Class | Purpose | Used By |
|-------|---------|---------|
| `PyMCModel` | Abstract base — provides fit/predict/score/calculate_impact contract | All Bayesian experiments (via subclasses) |
| `LinearRegression` | Standard linear model: `y ~ Normal(X·β, σ)` | ITS, DiD, RD, RKD, PrePostNEGD, PiecewiseITS, StaggeredDiD, PanelRegression |
| `WeightedSumFitter` | Dirichlet-weighted sum: `y ~ Normal(X·β, σ)` where `β ~ Dirichlet(1)` | SyntheticControl |
| `SoftmaxWeightedSumFitter` | Softmax-Normal simplex weights (alternative to Dirichlet) | SyntheticControl (alternative) |
| `SyntheticDifferenceInDifferencesWeightFitter` | Joint unit + time weight model for SDiD | SyntheticDifferenceInDifferences |
| `InstrumentalVariableRegression` | 2SLS with correlated errors (LKJ covariance or binary treatment) | InstrumentalVariable |
| `PropensityScore` | Logistic propensity model: `t ~ Bernoulli(logit⁻¹(X·b))` | InversePropensityWeighting |
| `BayesianBasisExpansionTimeSeries` | Trend + seasonality via pymc-marketing components (experimental) | InterruptedTimeSeries (alternative) |
| `StateSpaceTimeSeries` | State-space model via pymc-extras structural (experimental) | InterruptedTimeSeries (alternative) |

## Extension Guide

### Add a new experiment class

1. Create `causalpy/experiments/your_method.py`
2. Subclass `BaseExperiment`
3. Set `supports_ols`, `supports_bayes`, and optionally `_default_model_class`
4. Implement `__init__` calling `super().__init__(model=model)` then `_build_design_matrices()` → `algorithm()`
5. Implement `algorithm()` with the fit/predict/impact flow
6. Implement `_bayesian_plot()` and/or `_ols_plot()` (only for supported backends)
7. Implement `effect_summary()` using helpers from `causalpy.reporting`
8. Declare an explicit public `plot(*, ...)` method with kwarg-only signature that calls `self._render_plot(...)`
9. Export from `causalpy/experiments/__init__.py` and `causalpy/__init__.py`

### Add a new PyMC model

1. Add class to `causalpy/pymc_models.py` inheriting from `PyMCModel`
2. Implement `build_model(X, y, coords)` — must create `pm.Data("X", ...)`, `pm.Data("y", ...)`, a `pm.Deterministic("mu", ..., dims=["obs_ind", "treated_units"])`, and a likelihood named `"y_hat"`
3. Set `default_priors` dict with `Prior` objects
4. Optionally override `priors_from_data(X, y)` for data-adaptive priors
5. Optionally override `_data_setter(X)` if prediction requires custom data updates
6. If `fit()` signature differs from base (non-standard arguments), override it with `# type: ignore[override]`

### Add a new sklearn-compatible model

1. Create a class inheriting from both `ScikitLearnAdaptor` and sklearn's `LinearModel` + `RegressorMixin`
2. Implement `fit(X, y)` and `predict(X)` — store coefficients in `self.coef_` as 2D array
3. Alternatively, pass any fitted sklearn `RegressorMixin` instance — `create_causalpy_compatible_class()` will monkey-patch the adapter methods automatically

### Add a new plotting backend or report format

- For plots: override `_bayesian_plot()` / `_ols_plot()` in experiment subclass, or create a new dispatch in `_render_plot()`
- For reports: extend `causalpy/steps/report.py` (`GenerateReport` step produces HTML); the experiment's `generate_report()` method wraps this
- For table export: add a new adapter in `causalpy/maketables_adapters.py` implementing the `MaketablesAdapter` protocol

## Key Conventions and Gotchas

| Topic | Detail |
|-------|--------|
| **Intercept handling** | Patsy includes an intercept by default (`1 +` in formula). sklearn models must use `fit_intercept=False` because the intercept is already in the design matrix as a column of ones. |
| **treated_units is always 2D** | Even single-unit experiments use `treated_units=["unit_0"]`. y is always shape `(n_obs, n_treated)`. Never pass 1D y to a PyMC model. |
| **Impact uses mu, not y_hat** | `calculate_impact()` subtracts `posterior_predictive["mu"]` (expected value), not `["y_hat"]` (with observation noise). This gives cleaner effect estimates reflecting only parameter uncertainty. |
| **labels must align with coefficients** | `self.labels` comes from `X.design_info.column_names` and must match the `coeffs` dimension of the posterior `beta` variable exactly in order and length. |
| **maketables adapter dispatch** | `get_maketables_adapter(model)` mirrors the `isinstance` dispatch: `PyMCModel` → `PyMCMaketablesAdapter`, `RegressorMixin` → `SklearnMaketablesAdapter`. |
| **obs_ind index naming** | Experiments rename `data.index.name = "obs_ind"` early. All xarray dims use this name. PyMC coords key must be `"obs_ind"`. |
| **design_info for out-of-sample** | Store `_x_design_info` and `_y_design_info` from `dmatrices()`. Use `build_design_matrices([info], new_data)` for counterfactual prediction to preserve factor encoding. |
| **Eager fitting in __init__** | Most experiments run MCMC during `__init__`. There is no lazy `.fit()` — the experiment object is fully fitted upon construction. |
| **HDI_PROB default** | The project uses 0.94 (matching ArviZ default), NOT 0.95. `effect_summary()` defaults to `alpha=0.05` (95% HDI), which is independent of `HDI_PROB`. |
| **SyntheticControl multi-unit** | `SyntheticControl` loops over `treated_units` fitting one model per unit (via `_clone()`). Each unit gets its own `pre_pred`, `post_pred`, `impact`. |
| **SDiD cut-posterior** | Treatment effect tau is NOT estimated inside the MCMC model. Unit and time weights are sampled jointly, then tau is computed analytically via double-differencing the observed data with the weight posteriors. |
| **InstrumentalVariable non-standard fit** | `IVRegression.fit(X, Z, y, t, coords, priors, ...)` — does NOT follow the base `fit(X, y, coords)` signature. The experiment class handles this internally. |
| **create_causalpy_compatible_class mutates** | This function mutates the passed instance (adds methods), it does NOT create a new class or return a new instance. The name is misleading. |
| **Pipeline vs direct instantiation** | Experiments can be used standalone (just instantiate) or via `Pipeline` with steps. The pipeline adds `SensitivityAnalysis` and `GenerateReport` on top. |
