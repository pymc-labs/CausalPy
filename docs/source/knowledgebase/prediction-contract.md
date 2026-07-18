# Prediction contract for causal impact

CausalPy calculates Bayesian causal impact as the difference between observed outcomes and a counterfactual **expected outcome** in observed outcome units. Experiments call :meth:`~causalpy.pymc_models.PyMCModel.calculate_impact`, which subtracts ``posterior_predictive["mu"]`` from the observed ``y``. That subtraction only has a causal interpretation when ``mu`` is the conditional expectation :math:`E[Y \mid \text{parameters}, \text{covariates}]` on the **response scale**, excluding observation-level noise.

Gaussian models with an identity link make the linear predictor, conditional expectation, and outcome scale look interchangeable. For generalized linear models they are not. With a Poisson log link, the linear predictor :math:`\eta` is on the log-rate scale while counts live on the outcome scale; CausalPy needs :math:`\exp(\eta) = E[Y \mid \cdot]` in ``mu``, not :math:`\eta` itself. The same principle applies to Bernoulli/logit models, where ``mu`` must be a probability in :math:`(0, 1)`.

## Three prediction quantities

| Quantity | Typical name in CausalPy | Scale | Includes observation noise? | Used for impact? |
|----------|-------------------------|-------|----------------------------|------------------|
| Linear predictor / latent state | model-specific (e.g. ``eta``) | Link scale | No | No |
| Conditional expected outcome | ``mu`` | Outcome / response scale | No | **Yes** |
| Posterior predictive draw | ``y_hat`` | Outcome scale | Yes | No (by default) |

**Impact** uses the middle row: draw-level contrasts ``y - mu`` are computed in outcome units, then summarized. **Plots and effect summaries** inherit this convention unless a method documents a different estimand (for example link-scale regression coefficients).

Posterior transformations for nonlinear models must be applied **draw by draw** before contrasts or averages. In general, ``inverse_link(mean(eta))`` differs from ``mean(inverse_link(eta))``; g-computation averages unit-level potential outcomes on the response scale. See {doc}`estimands` for how this relates to empirical estimands and {doc}`../notebooks/g-computation-link-functions-pymc` for worked Poisson and logit examples.

## Backend obligations

Every Bayesian backend that participates in impact calculation must expose outcome-scale conditional expectations through ``mu`` in the object returned by ``predict()``:

| Backend | ``mu`` semantics today | Notes |
|---------|------------------------|-------|
| Native :class:`~causalpy.pymc_models.PyMCModel` subclasses (e.g. ``LinearRegression``) | ``Deterministic`` named ``mu`` in the PyMC graph | Identity-link Gaussian models satisfy the contract by construction. Custom subclasses must inverse-link before naming the node ``mu``. See {doc}`custom_pymc_models`. |
| :class:`~causalpy.pymc_models.StateSpaceTimeSeries` | ``mu`` aliases the smoothed expected observation | Gaussian observation model; ``mu`` and ``y_hat`` coincide up to naming. |
| :class:`~causalpy.pymc_forecast_models.PyMCForecastModel` | Upstream ``mu`` / ``mu_future`` latent passed to ``pymc_forecast.predict()`` | CausalPy treats these as outcome-scale expectations. For linked GLMs, pass the inverse-linked expectation as the latent until upstream exposes a dedicated expected-observation output (`pymc-forecast#52 <https://github.com/pymc-labs/pymc-forecast/issues/52>`_). ``StatespaceForecaster`` is rejected because upstream does not yet expose a separate noise-free expectation (`pymc-forecast#50 <https://github.com/pymc-labs/pymc-forecast/issues/50>`_). |
| scikit-learn adapters | Point predictions on the outcome scale | OLS backends return fitted values in ``y`` units; no separate ``mu`` draw dimension. |

sklearn backends compute impact as ``y_true - y_pred`` with the same outcome-scale requirement.

## Public name and compatibility

The public posterior predictive name remains ``mu`` for backward compatibility. The contract is **semantic**, not lexical: ``mu`` must denote the conditional expected outcome in observed units. A future explicit name such as ``expected_outcome`` may be added in a minor release once all built-in backends are audited; until then, custom model authors should treat ``mu`` as that quantity.

## Authoring checklist for custom PyMC models

1. Compute the linear predictor or latent state on the link scale if needed, but do not expose it as ``mu`` unless the outcome scale is the identity link.
2. Apply the inverse link draw by draw and register the result as a ``Deterministic`` named ``mu`` with dims ``["obs_ind", "treated_units"]``.
3. Keep ``y_hat`` as the likelihood / posterior predictive of the observed variable (including sampling noise where applicable).
4. Ensure ``predict()`` returns both ``mu`` and ``y_hat`` in ``posterior_predictive`` so :meth:`~causalpy.pymc_models.PyMCModel.calculate_impact` and :meth:`~causalpy.pymc_models.PyMCModel.score` behave consistently.

Fast regression tests in ``causalpy/tests/test_prediction_contract.py`` encode this contract for identity-link Gaussian, Poisson log-link, and Bernoulli logit models.

## Related issues

- CausalPy #1016 (this documentation and contract tests)
- CausalPy #1017 (g-computation tutorial notebook)
- `pymc-forecast#52 <https://github.com/pymc-labs/pymc-forecast/issues/52>`_ — explicit expected-observation outputs
- `pymc-forecast#50 <https://github.com/pymc-labs/pymc-forecast/issues/50>`_ — statespace expected observations
