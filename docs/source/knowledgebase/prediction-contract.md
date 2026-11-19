# Prediction contract for causal impact

CausalPy calculates Bayesian causal impact as the difference between observed outcomes and a counterfactual **expected outcome** in observed outcome units. Experiments call the model adapter's ``predict()``, which extracts ``posterior_predictive["mu"]``, and subtract that from the observed ``y``. That subtraction only has a causal interpretation when ``mu`` is the conditional expectation :math:`E[Y \mid \text{parameters}, \text{covariates}]` on the **response scale**, excluding observation-level noise.

Gaussian models with an identity link make the linear predictor, conditional expectation, and outcome scale look interchangeable. For generalized linear models they are not. With a Poisson log link, the linear predictor :math:`\eta` is on the log-rate scale while counts live on the outcome scale; CausalPy needs :math:`\exp(\eta) = E[Y \mid \cdot]` in ``mu``, not :math:`\eta` itself. The same principle applies to Bernoulli/logit models, where ``mu`` must be a probability in :math:`(0, 1)`.

## Three prediction quantities

| Quantity | Typical name in CausalPy | Scale | Includes observation noise? | Used for impact? |
| --- | --- | --- | --- | --- |
| Linear predictor / latent state | model-specific (e.g. ``eta``) | Link scale | No | No |
| Conditional expected outcome | ``mu`` | Outcome / response scale | No | **Yes** |
| Posterior predictive draw | ``y_hat`` | Outcome scale | Yes | No. Stored impact stays ``y - mu``. Outcome plots and plot-data exports can opt into ``y_hat`` with ``response_type="prediction"``. |

**Impact** uses the middle row: draw-level contrasts ``y - mu`` are computed in outcome units, then summarized. **Effect summaries** retain their documented estimand. Selecting a plotting target does not change stored results or a later summary call. Methods based on coefficients or structural contrasts retain those estimands in their effect panels as well.

## Optional observation-noise bands

Use the keyword-only ``response_type="expectation"`` (the default) for uncertainty in the conditional expected outcome, or ``response_type="prediction"`` for observation-level predictive outcomes. ``group="prior"`` versus ``group="posterior"`` independently selects the conditioning draws. Backends without observation-level predictive draws raise ``ValueError`` instead of substituting expectation intervals.

For example, after fitting an interrupted time series:

```python
fig, axes = result.plot(response_type="prediction", ci_prob=0.94)
predictive_data = result.get_plot_data(response_type="prediction", hdi_prob=0.94)
```

ITS and synthetic control show predictive outcome bands in both periods and **predictive difference intervals** for observed minus counterfactual ``y_hat``. Cumulative difference intervals sum each draw's trajectory before taking its interval; summing marginal interval endpoints gives a different and generally incorrect result. Predictive differences include observation noise. Their tail probabilities are not posterior probabilities about the expected causal effect, and wider outcome bands do not establish causal-identification assumptions. The {doc}`../notebooks/interrupted-time-series-pymc` example compares both targets.

Draws may be generated anew on each predictive call. The target, alignment and interval definition agree between plots and exports; exact finite-sample values need not agree across unseeded calls. For custom visualizations, use the exported values together rather than mixing results from separate random draws.

| Experiment/view | Meaning of ``response_type="prediction"`` |
| --- | --- |
| Interrupted time series, synthetic control, inherited UpDownGeoLift | Noisy outcomes and observed-minus-predictive difference panels/exports; stored expectation impacts remain unchanged. |
| Piecewise ITS | Noisy fitted/counterfactual outcome bands; structural-effect and cumulative-effect panels/exports retain their expectation contrast. |
| Difference-in-differences, pretest/posttest, regression discontinuity, regression kink | Noisy outcome bands; coefficient/threshold-effect estimates retain their existing estimand. |
| Staggered DiD event/cohort plots and event-time export | Aggregate observed-minus-predictive differences with the existing cohort eligibility rules and equal-tailed intervals. Export column names remain unchanged; ``DataFrame.attrs`` identifies the predictive-difference target and interval. Stored ATT tables and summaries remain expectation-based. |
| Panel regression ``get_plot_data`` | Noisy fitted outcomes with the existing 95% equal-tailed bounds. Its coefficient-only ``plot`` rejects predictive mode and directs callers to the outcome export or the existing ``plot_trajectories(interval_type="predictive")``. |
| Synthetic DiD | Unsupported: the weight model has no observation-level outcome likelihood. |
| Experimental interactive fixed effects | Unsupported: the reconstructed untreated trajectory exposes expectation draws, but no treated predictive-outcome sampler. |

The synthetic DiD, interactive fixed-effects and native state-space capabilities require model-specific follow-up work tracked in [issue #1459](https://github.com/pymc-labs/CausalPy/issues/1459). Panel coefficient plots, IV's unimplemented ``plot`` and IPW's dedicated treatment-effect/balance diagnostics do not depict noisy outcome trajectories; an outcome selector is not an appropriate replacement for their estimands. Augmented synthetic control currently provides fitting but no public plot/export view to extend.

Point-estimate backends have no observation-noise draws and reject predictive requests. Native state-space models currently alias ``mu`` and ``y_hat``; predictive mode rejects that alias rather than mislabeling expectation uncertainty. The optional forecasting backend supports predictive outcomes through its normalized ``y_hat`` output, with posterior conditioning only.

Bayesian ribbon plots name the outcome target, conditioning group, interval kind (HDI or ETI) and probability by default. Histogram and spaghetti plots can be labeled with ``show_response_annotation=True``; ``False`` suppresses the note. Coefficient and structural-effect panels retain their own labels. Summaries and persistence calculations remain expectation-based and do not accept an outcome-target selector.

Posterior transformations for nonlinear models must be applied **draw by draw** before contrasts or averages. In general, ``inverse_link(mean(eta))`` differs from ``mean(inverse_link(eta))``; g-computation averages unit-level potential outcomes on the response scale. See {doc}`estimands` for how this relates to empirical estimands. A worked tutorial notebook is tracked in `CausalPy#1017 <https://github.com/pymc-labs/CausalPy/issues/1017>`_.

## Backend obligations

Every Bayesian backend that participates in impact calculation must expose outcome-scale conditional expectations through ``mu`` in the object returned by ``predict()``:

| Backend | ``mu`` semantics today | Notes |
| --- | --- | --- |
| Native :class:`~causalpy.pymc_models.PyMCModel` subclasses (e.g. ``LinearRegression``) | ``Deterministic`` named ``mu`` in the PyMC graph | Identity-link Gaussian models satisfy the contract by construction. Custom subclasses must inverse-link before naming the node ``mu``. See {doc}`custom_pymc_models`. |
| :class:`~causalpy.pymc_models.StateSpaceTimeSeries` | ``mu`` aliases the smoothed expected observation | Gaussian observation model; ``mu`` and ``y_hat`` coincide up to naming, so ``response_type="prediction"`` is rejected rather than labeled as observation noise. |
| :class:`~causalpy.pymc_forecast_models.PyMCForecastModel` | Explicit upstream ``expected_observation`` / ``expected_observation_future`` when present; otherwise ``mu`` / ``mu_future`` | With pymc-forecast 0.3, pass the response-scale expectation to ``predict(..., expected_observation=...)``. The fallback latent must already be an outcome-scale expectation; CausalPy does not infer inverse links. ``StatespaceForecaster`` is rejected because upstream does not yet expose a separate noise-free expectation (`pymc-forecast#50 <https://github.com/pymc-labs/pymc-forecast/issues/50>`_). |
| scikit-learn adapters | Point predictions on the outcome scale | OLS backends return fitted values in ``y`` units; no separate ``mu`` draw dimension. |

sklearn backends compute impact as ``y_true - y_pred`` with the same outcome-scale requirement.

When upgrading from pymc-forecast 0.2 to 0.3, change model-builder callbacks from ``(h, covariates)`` to ``(covariates, data=None)`` and construct the horizon inside the callback with ``Horizon.from_data(covariates, data)``. Replace the 0.2 ``time_series`` pattern with ``innovations`` for independent innovations or ``markov_series`` for recursive states. Both in-sample and future predictions are normalized at the wrapper boundary into the same ``posterior_predictive.mu`` and ``y_hat`` containers; observation noise remains in ``y_hat`` only. The {doc}`../notebooks/interrupted-time-series-covid` example combines these building blocks.

## Score container

Every model adapter returns scores in one canonical ``pandas.Series``. Each treated unit has a required ``unit_{i}_r2`` entry in treated-unit order. Bayesian backends additionally provide ``unit_{i}_r2_std`` when posterior draws supply dispersion; point-estimate backends omit the standard-deviation entry rather than inventing uncertainty.

For multi-output sklearn models, CausalPy computes one :math:`R^2` per output instead of exposing sklearn's default uniform average. The shared container does not make the statistics identical: PyMC reports Bayesian :math:`R^2` and its posterior dispersion, while sklearn reports classical point-estimate :math:`R^2`. It only guarantees that consumers can locate each unit's score without knowing which backend produced it.

This is a user-visible change for sklearn experiments: ``experiment.result.score`` is a keyed ``pandas.Series`` rather than a bare float. Single-output users should read ``experiment.result.score["unit_0_r2"]``.

For direct adapter callers, ``SklearnModelAdapter.score()`` keyword arguments follow :func:`sklearn.metrics.r2_score` (for example ``sample_weight``), not the underlying estimator's ``score`` method. ``multioutput`` is fixed to ``"raw_values"`` so each treated unit keeps its own ``unit_{i}_r2`` entry.

## Canonical coefficient container

The model adapter's ``coefficients()`` method preserves coefficient uncertainty in a canonical :class:`xarray.DataArray` named ``"coefficients"``. Every supported backend returns dimensions ``("chain", "draw", "coeffs")`` in that order, with an optional trailing ``"treated_units"`` dimension when the model fits a separate coefficient vector per outcome. The ``coeffs`` coordinate contains the fitted design-matrix labels; ``treated_units`` contains the corresponding outcome labels.

PyMC adapters return all posterior draws, normalizing the built-in model families' native coefficient names and label dimensions at the adapter boundary. sklearn adapters embed fitted point estimates with singleton ``chain`` and ``draw`` dimensions. Consumers therefore use the same xarray operations for every backend: point estimates are ``coefficients.mean(("chain", "draw"))``, while intervals are only meaningful when ``coefficients.sizes["chain"] * coefficients.sizes["draw"] > 1``.

Coefficients remain on their model's parameter scale; unlike the prediction contract, CausalPy does not inverse-link them to the response scale. Models without design-matrix coefficients, including the ``pymc-forecast`` adapter, raise ``NotImplementedError``. Instrumental variables also retain a dedicated two-stage summary because its treatment and outcome equations are distinct coefficient surfaces rather than interchangeable backend containers.

## Public name and compatibility

The public posterior predictive name remains ``mu`` for backward compatibility. The contract is **semantic**, not lexical: ``mu`` must denote the conditional expected outcome in observed units. A future explicit name such as ``expected_outcome`` may be added in a minor release once all built-in backends are audited; until then, custom model authors should treat ``mu`` as that quantity.

## Authoring checklist for custom PyMC models

1. Compute the linear predictor or latent state on the link scale if needed, but do not expose it as ``mu`` unless the outcome scale is the identity link.
2. Apply the inverse link draw by draw and register the result as a ``Deterministic`` named ``mu`` with dims ``["obs_ind", "treated_units"]``.
3. Keep ``y_hat`` as the likelihood / posterior predictive of the observed variable (including sampling noise where applicable).
4. Ensure ``predict()`` returns both ``mu`` and ``y_hat`` in ``posterior_predictive`` so the model adapter's ``predict()`` and :meth:`~causalpy.pymc_models.PyMCModel.score` behave consistently.

Fast regression tests in ``causalpy/tests/test_prediction_contract.py`` encode this contract for identity-link Gaussian, Poisson log-link, and Bernoulli logit models.

## Related issues

- CausalPy #1016 (this documentation and contract tests)
- CausalPy #1017 (g-computation tutorial notebook)
- `pymc-forecast#52 <https://github.com/pymc-labs/pymc-forecast/issues/52>`_ — explicit expected-observation outputs
- `pymc-forecast#50 <https://github.com/pymc-labs/pymc-forecast/issues/50>`_ — statespace expected observations
