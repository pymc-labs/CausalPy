# Falsification Tests

Falsification tests do not prove causality. They ask whether specific alternative explanations make predictions that are contradicted by the data.

## Pre-Treatment Placebo

Alternative tested: the effect existed before treatment, suggesting confounding, anticipation, or a broken counterfactual.

```python
check = cp.checks.PreTreatmentPlaceboCheck()
result = check.run(experiment)
```

Pass: no meaningful pre-treatment effect. Fail: the claim needs a stronger explanation or different design.

## Placebo In Time

Alternative tested: the model detects fake effects in periods where no intervention occurred.

```python
check = cp.checks.PlaceboInTime(
    n_folds=4,
    experiment_factory=experiment_factory,
)
result = check.run(experiment)
```

Compare the actual effect to the learned null distribution. Interpret `p_effect_outside_null` and effect sizes in context.

## Outcome Falsification

Alternative tested: something correlated with treatment affects outcomes broadly, including outcomes that should not respond to treatment.

```python
check = cp.checks.OutcomeFalsification(
    outcome_column="y",
    falsification_column="placebo_y",
)
result = check.run(experiment)
```

Choose placebo outcomes using domain knowledge. A poor placebo outcome weakens the test.

## Leave One Out

Alternative tested: the result depends on a single donor, unit, or observation.

```python
check = cp.checks.LeaveOneOut()
result = check.run(experiment)
```

Large swings after dropping one unit indicate fragility.

## Placebo In Space

Alternative tested: similar effects appear in untreated units, suggesting common shocks or model artifacts.

```python
check = cp.checks.PlaceboInSpace(n_placebo_units=5)
result = check.run(experiment)
```

This is most natural when multiple untreated units are available.

## Bandwidth Sensitivity

Alternative tested: an RD or RK result depends on one arbitrary bandwidth.

```python
check = cp.checks.BandwidthSensitivity(
    bandwidth_start=0.1,
    bandwidth_stop=1.0,
    bandwidth_step=0.1,
)
result = check.run(experiment)
```

Stable signs and magnitudes across reasonable bandwidths strengthen the claim.

## Prior Sensitivity

Alternative tested: the Bayesian result is driven by prior choices rather than data.

```python
check = cp.checks.PriorSensitivity(
    prior_range={"sigma": [0.5, 1.0, 2.0, 5.0]}
)
result = check.run(experiment)
```

Use plausible prior ranges based on the outcome scale and domain.

## McCrary Density Test

Alternative tested: units manipulate the running variable around an RD cutoff.

```python
check = cp.checks.McCraryDensityTest()
result = check.run(experiment)
```

Suspicious bunching at the threshold weakens the continuity argument.

## Convex Hull Check

Alternative tested: a synthetic-control counterfactual extrapolates beyond donor support.

```python
check = cp.checks.ConvexHullCheck()
result = check.run(experiment)
```

Poor support means the synthetic control may be a model extrapolation rather than a credible comparison.

## Persistence Check

Alternative tested: the effect is a short-lived artifact of the chosen window or reverses quickly.

```python
check = cp.checks.PersistenceCheck()
result = check.run(experiment)
```

Interpret persistence relative to the theory of how the intervention should work.
