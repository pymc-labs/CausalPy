# Architecture

This document describes how CausalPy is structured internally. It is intended for contributors — human or AI — who need to understand where things live, how the pieces fit together, and where to add new functionality. For user-facing documentation, see the [docs site](https://causalpy.readthedocs.io/).

## Design overview

CausalPy implements quasi-experimental causal inference methods across two statistical backends (PyMC for Bayesian, scikit-learn for frequentist). The codebase is organized in layers:

```
Experiments  →  Models  →  Reporting
    ↓              ↓           ↑
 Pipeline  →  Steps  →  Checks / Sensitivity
```

- **Experiments** define the causal design (DiD, ITS, SC, etc.) and orchestrate fitting, plotting, and summarization.
- **Models** handle statistical estimation — either Bayesian (PyMC) or frequentist (scikit-learn).
- **Reporting** generates narrative effect summaries with HDI/ROPE (Bayesian) or CI/p-value (OLS) output.
- **Pipeline** chains steps (estimate → sensitivity → report) with a validate-before-run guarantee.
- **Checks** implement sensitivity analyses (placebo tests, bandwidth sensitivity, leave-one-out, etc.).

## Directory layout

```
causalpy/
├── __init__.py                  # Public API surface
├── experiments/                 # Quasi-experimental design classes
│   ├── base.py                  # BaseExperiment ABC
│   ├── diff_in_diff.py          # DifferenceInDifferences
│   ├── staggered_did.py         # StaggeredDifferenceInDifferences
│   ├── interrupted_time_series.py
│   ├── piecewise_its.py         # PiecewiseITS (step/ramp formulas)
│   ├── synthetic_control.py     # SyntheticControl
│   ├── regression_discontinuity.py
│   ├── regression_kink.py
│   ├── instrumental_variable.py
│   ├── inverse_propensity_weighting.py
│   └── prepostnegd.py           # PrePostNEGD
├── pymc_models.py               # Bayesian models (extend pm.Model)
├── skl_models.py                # Scikit-learn integration
├── reporting.py                 # EffectSummary and narrative generation
├── pipeline.py                  # Pipeline, Step protocol, PipelineContext
├── steps/                       # Pipeline step implementations
│   ├── estimate.py              # EstimateEffect
│   ├── sensitivity.py           # SensitivityAnalysis + default registry
│   └── report.py                # GenerateReport (Jinja2 HTML)
├── checks/                      # Sensitivity / diagnostic checks
│   ├── base.py                  # Check protocol, CheckResult, clone_model
│   ├── placebo_in_time.py
│   ├── prior_sensitivity.py
│   ├── bandwidth_sensitivity.py
│   ├── convex_hull.py
│   ├── leave_one_out.py
│   ├── mccrary_density.py
│   ├── persistence.py
│   ├── placebo_in_space.py
│   └── pre_treatment_placebo.py
├── transforms.py                # Patsy stateful transforms (step, ramp)
├── variable_selection_priors.py # Spike-and-slab, horseshoe priors
├── reporting.py                 # EffectSummary and prose generation
├── custom_exceptions.py         # FormulaException, DataException, etc.
├── date_utils.py                # Date axis formatting helpers
├── data/                        # Datasets
│   ├── datasets.py              # load_data() dispatcher
│   └── simulate_data.py         # Synthetic data generators
├── tests/                       # pytest suite
└── version.py
```

## The experiment layer

All experiment classes inherit from `BaseExperiment` (`causalpy/experiments/base.py`), which is an ABC that enforces a contract through two class-level flags:

- **`supports_bayes: bool`** — whether the experiment works with PyMC models.
- **`supports_ols: bool`** — whether the experiment works with scikit-learn models.

Construction validates the model type against these flags and raises `ValueError` if mismatched. If `model=None` is passed, the experiment instantiates `_default_model_class` (always a PyMC model class).

### Public interface

| Method | Role |
|---|---|
| `plot(...)` | Dispatches to `_bayesian_plot()` or `_ols_plot()` based on model type |
| `get_plot_data(...)` | Dispatches to `get_plot_data_bayesian()` or `get_plot_data_ols()` |
| `effect_summary(...)` | Abstract — returns an `EffectSummary` with table and narrative |
| `generate_report(...)` | Builds a pipeline context and runs `GenerateReport` to produce HTML |
| `print_coefficients(...)` | Delegates to `self.model.print_coefficients()` |
| `idata` | Pass-through to `self.model.idata` (Bayesian only) |

Subclasses override the `_bayesian_plot` / `_ols_plot` and `get_plot_data_*` methods only for the backends they support. Estimation is performed during construction (each experiment's `__init__` calls `self.model.fit(...)` internally), not via a separate `fit()` call.

### Experiment catalog

| Class | Method | OLS | Bayes | Default model |
|---|---|---|---|---|
| `DifferenceInDifferences` | Classic DiD | yes | yes | `LinearRegression` |
| `StaggeredDifferenceInDifferences` | Staggered DiD (BJS imputation) | yes | yes | `LinearRegression` |
| `InterruptedTimeSeries` | ITS (2- or 3-period) | yes | yes | `LinearRegression` |
| `PiecewiseITS` | Segmented ITS with step/ramp | yes | yes | `LinearRegression` |
| `SyntheticControl` | Synthetic control (donor weights) | yes | yes | `WeightedSumFitter` |
| `RegressionDiscontinuity` | Sharp RD | yes | yes | `LinearRegression` |
| `RegressionKink` | Regression kink | — | yes | `LinearRegression` |
| `InstrumentalVariable` | IV (two-stage) | — | yes | `InstrumentalVariableRegression` |
| `InversePropensityWeighting` | IPW (propensity + weighting) | — | yes | `PropensityScore` |
| `PrePostNEGD` | Pretest/posttest NEGD | — | yes | `LinearRegression` |

## The model layer

### PyMC models (`causalpy/pymc_models.py`)

`PyMCModel` subclasses `pm.Model` and adds a scikit-learn-like interface. Subclasses implement `build_model(X, y, coords)` to define the generative model, where `X` and `y` are `xarray.DataArray` with dims `["obs_ind", "coeffs"]` and `["obs_ind"]` (or `["obs_ind", "treated_units"]`).

**Core methods:**

| Method | What it does |
|---|---|
| `fit(X, y, coords)` | Merges priors, calls `build_model`, then `pm.sample` + posterior/prior predictive |
| `predict(X)` | Sets data via `_data_setter`, runs posterior predictive, returns `y_hat` and `mu` |
| `score(X, y)` | Bayesian R² via ArviZ per treated unit |
| `calculate_impact(y_true, y_pred)` | `y_observed - E[y_counterfactual]` from posterior `mu` |
| `print_coefficients(labels)` | Posterior summaries of `beta` and `sigma` with HDI |

**Concrete models:** `LinearRegression`, `WeightedSumFitter` (Dirichlet SC), `SoftmaxWeightedSumFitter`, `InstrumentalVariableRegression`, `PropensityScore`, `BayesianBasisExpansionTimeSeries`, `StateSpaceTimeSeries`.

### Scikit-learn models (`causalpy/skl_models.py`)

Any scikit-learn `RegressorMixin` can be used with CausalPy experiments. `BaseExperiment.__init__` calls `create_causalpy_compatible_class(model)` to attach CausalPy-specific methods (`calculate_impact`, `calculate_cumulative_impact`, `print_coefficients`) from the `ScikitLearnAdaptor` mixin onto the model instance.

`WeightedProportion` is a built-in sklearn-style model for frequentist synthetic control (nonneg weights summing to 1 via constrained optimization).

## The reporting layer (`causalpy/reporting.py`)

`EffectSummary` is a dataclass with two fields: `table` (a `pd.DataFrame` of numeric results) and `text` (a prose narrative).

The module contains separate generation paths for Bayesian and OLS results:

- **Bayesian**: HDI intervals, tail probabilities, optional ROPE analysis. Uses `arviz.hdi` on `xarray.DataArray` posteriors.
- **OLS**: Confidence intervals and p-values from standard errors.
- **Time-varying effects** (ITS, SC): Point-by-point `y_observed - y_counterfactual` with cumulative and relative impact.
- **Scalar effects** (DiD, RD, RKink): Single treatment effect estimate with uncertainty.

`_detect_experiment_type` inspects result attributes to route to the correct summary generator.

## The pipeline system (`causalpy/pipeline.py`)

The pipeline provides a structured way to chain estimation, sensitivity analysis, and reporting.

**Core types:**

- **`Step` (Protocol)**: Must implement `validate(context)` and `run(context)`. All steps are validated before any are run, so misconfiguration fails fast.
- **`PipelineContext`**: Mutable dataclass carrying `data`, `experiment`, `effect_summary`, `sensitivity_results`, and `report` through the pipeline.
- **`PipelineResult`**: Immutable snapshot returned by `Pipeline.run()`.
- **`Pipeline(data, steps)`**: Validates all steps, then runs them sequentially.

**Built-in steps** (in `causalpy/steps/`):

| Step | Purpose |
|---|---|
| `EstimateEffect` | Instantiates an experiment class with the pipeline data |
| `SensitivityAnalysis` | Runs a set of `Check` instances against the fitted experiment |
| `GenerateReport` | Renders Jinja2 HTML with plots, effect tables, and sensitivity results |

## The checks system (`causalpy/checks/`)

Checks implement sensitivity analyses and diagnostic tests. Each check follows the `Check` protocol:

- **`applicable_methods`**: Class attribute — the set of experiment types this check applies to.
- **`validate(experiment)`**: Lightweight config validation.
- **`run(experiment, context)`**: Executes the check and returns a `CheckResult`.

`CheckResult` is a dataclass with `check_name`, `passed` (tri-state), `table`, `text`, `figures`, and `metadata`.

| Check | Applies to |
|---|---|
| `PlaceboInTime` | ITS, SyntheticControl |
| `PriorSensitivity` | All Bayesian experiments |
| `BandwidthSensitivity` | RD, RegressionKink |
| `ConvexHullCheck` | SyntheticControl |
| `LeaveOneOut` | SyntheticControl |
| `McCraryDensityTest` | RD |
| `PersistenceCheck` | ITS |
| `PlaceboInSpace` | SyntheticControl |
| `PreTreatmentPlaceboCheck` | StaggeredDiD |

Checks that need to re-fit models use `clone_model()` which calls `model._clone()` for PyMC models (avoiding `deepcopy` issues with PyMC's model context).

## The formula interface (`causalpy/transforms.py`)

CausalPy uses [patsy](https://patsy.readthedocs.io/) for formula parsing. The `transforms` module provides two patsy stateful transforms for use in `PiecewiseITS` formulas:

- **`step(x, threshold)`**: Returns 1 if `x >= threshold`, else 0. Handles datetime indices by converting to days from training minimum.
- **`ramp(x, threshold)`**: Returns `max(0, x - threshold)`. For datetime, the slope unit is change per day.

These transforms allow formulas like `y ~ 1 + t + step(t, 50) + ramp(t, 50)` to model level shifts and slope changes at known intervention points.

## The data layer (`causalpy/data/`)

`load_data(name)` is the single entry point for loading example datasets. It dispatches to:

- **Synthetic datasets**: Generated on-the-fly by functions in `simulate_data.py`. Keys include `"did"`, `"rd"`, `"sc"`, `"its"`, `"its simple"`, `"anova1"`, `"geolift1"`, etc.
- **Real-world datasets**: Bundled CSV files (e.g. `banks.csv`, `brexit.csv`, `nhefs.csv`, `lalonde.csv`). Loaded from the package's `data/` directory.

## Support modules

- **`variable_selection_priors.py`**: Spike-and-slab and horseshoe priors for variable selection, built on `pymc_extras.prior.Prior`. Includes post-fit helpers for inclusion probabilities and shrinkage factors.
- **`custom_exceptions.py`**: `FormulaException`, `DataException`, `BadIndexException` for clear error messages.
- **`date_utils.py`**: Matplotlib date axis formatting helpers.

## Tests (`causalpy/tests/`)

Tests are organized by area:

- **Integration tests**: `test_integration_pymc_examples.py`, `test_integration_skl_examples.py` — end-to-end experiment runs with minimal sampling.
- **Model tests**: `test_pymc_models.py`, `test_variable_selection_priors.py`.
- **Experiment-specific**: `test_staggered_did.py`, `test_instrumental_variable.py`, `test_piecewise_its.py`, etc.
- **Reporting/pipeline**: `test_reporting.py`, `test_pipeline.py`, `test_generate_report.py`.
- **Checks**: `test_method_specific_checks.py`, `test_cross_cutting_checks.py`, `test_placebo_in_time.py`.
- **Data**: `test_data_loading.py`, `test_synthetic_data.py`.

All tests use pytest-style functions (no unittest classes). Tests involving PyMC sampling use minimal `sample_kwargs` to keep the suite fast.

## Extension points

### Adding a new experiment

1. Create a new file in `causalpy/experiments/`.
2. Subclass `BaseExperiment`, setting `supports_ols` and `supports_bayes`.
3. Set `_default_model_class` to a sensible PyMC model.
4. Implement `__init__` (build design matrices, fit model), `effect_summary`, and the appropriate `_bayesian_plot` / `_ols_plot` methods.
5. Export the class from `causalpy/experiments/__init__.py` and `causalpy/__init__.py`.
6. Add tests and a documentation notebook.

### Adding a new PyMC model

1. Subclass `PyMCModel` in `causalpy/pymc_models.py`.
2. Implement `build_model(X, y, coords)`.
3. Optionally override `priors_from_data` for data-driven priors, or `_data_setter` for prediction-time data updates.

### Adding a new check

1. Create a new file in `causalpy/checks/`.
2. Implement the `Check` protocol: set `applicable_methods`, implement `validate` and `run`.
3. Export from `causalpy/checks/__init__.py`.
4. Optionally register as a default check via `register_default_check`.

## Keeping this document current

Update `ARCHITECTURE.md` when a PR introduces:

- A new experiment class, model, or check
- A new pipeline step
- Changes to the `BaseExperiment` contract or model interface
- Structural reorganization of modules

Minor bug fixes, documentation updates, and test additions do not require updates.
