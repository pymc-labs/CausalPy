# Release notes

This page records user-visible changes, with an emphasis on anything that can
break existing code. The auto-generated, commit-level history for each tagged
release lives on the
[GitHub releases](https://github.com/pymc-labs/CausalPy/releases) page; this
page is the curated, human-written companion to it.

## 1.0.0 (unreleased) — PyMC 6 migration

This is a **major, breaking release**. It migrates CausalPy off the PyMC 5 stack and onto PyMC 6 / PyTensor 3 / ArviZ 1.x; the docs-only PyMC-Marketing transition snapshot is pinned separately in `docs/requirements.txt`. Read the breaking-change section below before upgrading; several public signatures and default behaviours changed.

### Declared dependency ranges

- **Python `>=3.12`** (raised from 3.11). The dependency stack currently resolves on Python 3.12–3.14 (PyTensor caps Python `<3.15`). CI exercises 3.12 and 3.14.
- **pymc `>=6.0.1,<7`** (previously the PyMC 5 series). PyMC metadata determines its compatible PyTensor patch range. The pinned docs-only PyMC-Marketing transition snapshot narrows only a combined docs installation to `pymc>=6.0.1,<6.1`; base installs that omit `docs/requirements.txt` can select later compatible PyMC 6 minors.
- **pytensor `>=3,<4`**. Its selected minor and patch version are paired by PyMC metadata.
- **arviz `>=1.1,<2`** — the ArviZ 0.x → 1.x jump. `arviz.InferenceData` is no longer available as a usable class — accessing it emits a `MigrationWarning` (`"arviz.InferenceData is no longer available on the arviz package"`) — and ArviZ now uses xarray's `DataTree` for the same role.
- **pandas `>=2.3,<4`** — pandas 2.3 through the 3.x line are supported and are exercised as separate CI legs.
- **pymc-extras `>=0.11`**. This is the first release that declares PyMC 6 support and provides the structural state-space API used by CausalPy.
- **numba is now effectively a hard (transitive) dependency.** It is PyTensor 3's default compilation backend and is pulled in through the PyTensor/PyMC dependency chain. See [Runtime characteristics](#runtime-characteristics).

These declared ranges are in `pyproject.toml`.

### Breaking changes

#### Inference results are now `xarray.DataTree`, not `arviz.InferenceData`

Under ArviZ 1.x, `pm.sample()` — and therefore the `.idata` attribute on
CausalPy's PyMC models and experiments — is an `xarray.DataTree` rather than an
`arviz.InferenceData`. CausalPy's own type annotations reflect this (for
example `causalpy/pymc_forecast_models.py` annotates `self.idata: xr.DataTree
| None`).

If you wrote code against the old `InferenceData` object you may need to update
it. In particular:

- `InferenceData.extend()` no longer exists; groups are combined with
  `DataTree` operations instead.
- Constructing results via `az.InferenceData(group=ds)` no longer works.
- Accessing a group returns a `DataTree` node, which does not expose the
  group-level `.assign_coords` / `.rename` shortcuts that `InferenceData`
  groups had.

These are ArviZ/PyMC-level changes; they surface through any CausalPy object
that exposes an `idata`.

#### Public method signatures were narrowed (no more silent `**kwargs`)

Public methods across the experiment classes had their `*args` / `**kwargs`
catch-alls removed so that the supported keyword arguments are explicit
(PR #1107). A call that passes a keyword the method does not declare now raises
`TypeError`, where previously the argument was silently accepted and ignored.
This narrowing is enforced in CI (`scripts/audit_public_signatures.py` and
`causalpy/tests/test_public_signatures.py`). The two most user-visible
consequences are `get_plot_data` and `effect_summary`, below.

#### `get_plot_data` is now keyword-only, and absent on several experiments

`get_plot_data` used to be defined generically on `BaseExperiment` (as
`get_plot_data(self, *args, **kwargs)`, which raised `NotImplementedError` for
experiments that did not override it). That generic method was removed. As a
result:

- On the four experiments that implement it with an `hdi_prob` argument,
  `get_plot_data` is now **keyword-only**:
  - `InterruptedTimeSeries` (`causalpy/experiments/interrupted_time_series.py`)
  - `PiecewiseITS` (`causalpy/experiments/piecewise_its.py`)
  - `StaggeredDifferenceInDifferences` (`causalpy/experiments/staggered_did.py`)
  - `SyntheticControl` (`causalpy/experiments/synthetic_control.py`)

  These previously accepted `hdi_prob` positionally, so a call such as
  `experiment.get_plot_data(0.9)` now raises `TypeError`. Use the keyword form,
  `experiment.get_plot_data(hdi_prob=0.9)`.
- On `PanelRegression` (`causalpy/experiments/panel_regression.py`),
  `get_plot_data` is now **argument-less**. It previously accepted `**kwargs`
  (which were ignored), so passing any argument now raises `TypeError`. A bare
  `experiment.get_plot_data()` call still works.
- Because there is no longer a `BaseExperiment.get_plot_data`, calling
  `get_plot_data` on an experiment that does not define it now raises
  `AttributeError` (previously it raised `NotImplementedError`). The seven
  experiments in this situation are:
  1. `DifferenceInDifferences`
  2. `InstrumentalVariable`
  3. `InversePropensityWeighting`
  4. `PrePostNEGD`
  5. `RegressionKink`
  6. `RegressionDiscontinuity`
  7. `SyntheticDifferenceInDifferences`

  (This split — five experiments implement `get_plot_data`, seven do not — is
  pinned by `causalpy/tests/test_public_signatures.py`.)

#### `effect_summary(alpha=...)` raises `TypeError` instead of `NotImplementedError`

On the three experiments that do not implement a unified effect summary —
`PanelRegression` (`causalpy/experiments/panel_regression.py`),
`InstrumentalVariable` (`causalpy/experiments/instrumental_variable.py`), and
`InversePropensityWeighting`
(`causalpy/experiments/inverse_propensity_weighting.py`) — `effect_summary` is
now defined as `def effect_summary(self) -> NoReturn` with no parameters. A
bare `experiment.effect_summary()` call still raises `NotImplementedError` as
before. But because the method no longer declares an `alpha` (or any other)
parameter, a call such as `experiment.effect_summary(alpha=0.05)` now raises
`TypeError` *before* the body runs, whereas previously the same call reached
the body and raised `NotImplementedError`. The operation remains unsupported on
these three experiments; only the exception type for the `alpha=` form changed.

#### `SyntheticControl` default prior changed

`SyntheticControl` now derives its observation-noise `sigma` prior from each
treated unit's pre-treatment spread (a data-scaled prior), controlled by the
new `auto_scale_sigma` argument, which defaults to `True`
(`causalpy/experiments/synthetic_control.py`; PR #1106). This changes the
default posterior relative to earlier releases. To restore the previous fixed
`HalfNormal(1)` prior — for example when reproducing an analysis run against an
older release — pass `SyntheticControl(..., auto_scale_sigma=False)`.

#### `PlaceboInTime` abstains when its null cannot be identified

`PlaceboInTime` now returns `INCONCLUSIVE` (`passed=None`) rather than a `SUPPORTED` or `NOT-SUPPORTED` verdict when fewer than `MIN_USABLE_FOLDS` (2) usable placebo folds complete or the completed folds have a non-positive/non-finite between-fold spread. Because it does not build a hierarchical null in those cases, the result metadata has no `null_samples` or `p_effect_outside_null`. The guard catches only exact degeneracy: a tiny-but-positive between-fold spread on a large-scale series remains eligible for a verdict. Choosing a relative threshold is an unresolved modelling decision.

#### `InstrumentalVariable` samples with `cores=1`

The instrumental-variable model (an `MvNormal` with an `LKJCholeskyCov` prior)
can crash under multiprocess ("fork") sampling on an affected platform/version
combination (observed on macOS arm64 with Python 3.14 and PyMC 6.0.1–6.2.0) —
the worker dies natively rather than raising a Python exception. To keep the
model usable, CausalPy forces `cores=1` for it when `cores` is not otherwise
specified (`causalpy/pymc_models.py:1330-1332`). IV sampling therefore does not
parallelise across cores by default. This is a temporary mitigation: the
upstream bug is tracked at
[pymc-devs/pymc#8377](https://github.com/pymc-devs/pymc/issues/8377), and
removal of the workaround is tracked in CausalPy issue #1067.

#### Lazy experiment lifecycle: `configure` → optional prior checks → `fit()`

Experiment constructors no longer run inference. `__init__` validates input and builds design matrices only; posterior inference happens through an explicit `fit()` — which returns the fitted experiment, so existing call sites migrate with one appended token:

```python
# Before (0.x / eager)
result = cp.InterruptedTimeSeries(data, treatment_time=t0, formula="y ~ 1 + t")
result.plot()

# After (1.0 / lazy)
result = cp.InterruptedTimeSeries(data, treatment_time=t0, formula="y ~ 1 + t").fit()
result.plot()
```

The optional prior phase runs before MCMC when you want prior predictive checks: `exp.sample_prior_predictive()` (draws controlled by the model's `prior_sample_kwargs`, default 500), then `exp.plot(group="prior")`, `exp.effect_summary(group="prior")`, and `exp.get_plot_data(group="prior")`. Prior-group plots render a reduced panel set (counterfactual vs observations only) and prior effect summaries are worded as plausibility checks, not causal claims. `build()` constructs the PyMC graph without sampling so the spec can be inspected (`pm.model_to_graphviz(exp.model)`); both samplers auto-call it.

Draw-derived results moved off the experiment object into per-group bundles: `exp.result` (posterior group) and `exp.prior_result` (prior group). The old flat attributes are removed: `pre_pred` → `result.predictions_pre`, `post_pred` → `result.predictions_post`, `pre_impact` → `result.impact_pre`, `post_impact` → `result.impact_post`, `post_impact_cumulative` → `result.impact_post_cumulative`, `score` → `result.score`; DiD/PrePostNEGD's `causal_impact`, RD's `discontinuity_at_threshold`, RK's `gradient_change`, StaggeredDiD's `att_group_time_`/`att_event_time_`/`y_pred`/`hdi_prob_`, and SDID's weight-derived attributes live on the corresponding bundle fields. Read methods raise `GroupNotSampleedException` (new; exported from `causalpy`) naming the missing call instead of failing deep inside plotting or reporting. Attempting the prior phase on backends without one raises `PriorPredictiveNotSupportedException`.

Re-running a phase overwrites only its own draws: a second `fit()` replaces the posterior (emitting a warning) and preserves prior state. Assigning a new model — also the documented way to revise priors, replacing the never-shipped `set_priors()` — resets all results, because graph identity is the model instance. Third-party experiment subclasses that overrode `algorithm()` must migrate to `_fit_inputs()` + `_finalize(group)`; see `ARCHITECTURE.md`.

### Behaviour that intentionally did *not* change

#### The ArviZ default-interval change is a no-op for CausalPy

ArviZ 1.x changed *its own* default summary interval from a 0.94 highest-density
interval (HDI) to a 0.89 equal-tailed interval (ETI). This does **not** change
any CausalPy output. CausalPy always passes an explicit interval probability
when it calls ArviZ — its HDI helper calls `az.hdi(..., prob=HDI_PROB)`, with
the project-wide `HDI_PROB = 0.94` defined in `causalpy/constants.py` — so
nothing falls through to ArviZ's new default. Interval widths in effect
summaries, plots, and `get_plot_data` are identical to the previous release.
(This is separate from the fact that different reports use different nominal
levels — some effect summaries report a 95% interval, and
`PanelRegression.get_plot_data` reports 95% quantile bounds — none of which
changed in this release.)

### Runtime characteristics

See also the [runtime notes in the README](https://github.com/pymc-labs/CausalPy#runtime-notes).

- **numba first-compile latency.** Because numba is PyTensor 3's default
  backend, the first sample from a freshly defined model incurs a noticeable
  compilation delay before sampling begins.
- **Native worker crashes replace Python tracebacks.** When a numba-compiled
  sampler worker fails, the failure can present as a native crash (or a stalled
  worker) rather than a Python traceback. A wedged worker can therefore look
  like a silent hang.
- **nutpie changes sampling defaults if installed.** If a recent enough
  `nutpie` is installed, PyMC selects it as the default NUTS sampler unless you
  pass an explicit `nuts_sampler` — so *installing* nutpie is itself enough to
  change how `pm.sample()` behaves. nutpie also applies its own tuning-step
  default rather than PyMC's built-in NUTS default of 1000 tune steps (with
  `tune=None`, PyMC forwards the request straight to nutpie rather than
  resolving it to 1000). The migration investigation reported nutpie's default
  as 400 tune steps; that exact number was not re-verified for this release
  (nutpie is optional and was not installed in the release environment), so
  treat 400 as indicative and confirm against your installed nutpie version if
  it matters for your run.

### Version

This release is **1.0.0**: the migration is backwards-incompatible, so the major version increments. The version is single-sourced from `causalpy/version.py`, and `pyproject.toml` reads it dynamically, so `causalpy.__version__` and the installed distribution metadata cannot drift apart.
