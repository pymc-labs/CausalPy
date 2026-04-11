# Falsification Tests

The logic: if an alternative explanation is true, it makes a specific, testable prediction. Run the test. If the prediction fails, rule out the alternative. If it holds, the causal claim is weakened.

## Test 1: Pre-Treatment Placebo

**Alternative being tested:** The effect existed before treatment (confounding or pre-existing trend).

**Logic:** If the treatment caused the effect, there should be no effect before treatment time. If we find one, something else is driving the result.

```python
check = cp.checks.PreTreatmentPlaceboCheck()
result = check.run(experiment)
# PASS = no pre-treatment effects (good)
# FAIL = effects before treatment (threat not ruled out)
```

## Test 2: Placebo-in-Time

**Alternative being tested:** The model picks up noise or structural patterns, not a real treatment effect.

**Logic:** Shift the treatment time to periods when no treatment occurred. Build a null distribution of "fake effects." Compare the actual effect to this null. If the actual effect is indistinguishable from the null, it's likely noise.

```python
check = cp.checks.PlaceboInTime(
    n_folds=4,
    experiment_factory=my_factory,  # rebuilds experiment at different times
)
result = check.run(experiment)
p = result.metadata["p_effect_outside_null"]
# High p = actual effect is outside the null (good)
# Low p = actual effect looks like noise (bad)
```

**With Bayesian assurance (power analysis):**
```python
import preliz as pz

check = cp.checks.PlaceboInTime(
    n_folds=4,
    experiment_factory=my_factory,
    expected_effect_prior=pz.maxent(pz.Normal(), lower=60, upper=120, mass=0.95),
    rope_half_width=50.0,
    random_seed=42,
)
result = check.run(experiment)
ar = result.metadata["assurance_result"]
print(f"Power to detect real effect: {ar.true_positive_rate:.1%}")
```

## Test 3: Outcome Falsification

**Alternative being tested:** The treatment is affecting things it shouldn't (suggesting confounding).

**Logic:** Run the same analysis on an outcome that the treatment should NOT affect. If we find an effect there too, something else is driving results.

```python
check = cp.checks.OutcomeFalsification(
    outcome_column="y",              # real outcome
    falsification_column="placebo_y" # outcome that should be unaffected
)
result = check.run(experiment)
# PASS = no effect on placebo outcome (good)
# FAIL = effect on placebo outcome (confounding likely)
```

## Test 4: Leave-One-Out

**Alternative being tested:** The result depends heavily on a single unit or observation (fragility).

**Logic:** Remove one observation at a time and refit. If results change dramatically, the finding is fragile.

```python
check = cp.checks.LeaveOneOut()
result = check.run(experiment)
```

## Test 5: Placebo-in-Space

**Alternative being tested:** The effect is not specific to the treated unit — it appears everywhere (common shock).

**Logic:** Apply the treatment to untreated units. If they also show "effects," the treated unit's result may not be causal.

```python
check = cp.checks.PlaceboInSpace(n_placebo_units=5, n_leads=3)
result = check.run(experiment)
```

## Test 6: Bandwidth Sensitivity (RD/RK)

**Alternative being tested:** The result is sensitive to the bandwidth choice (model specification sensitivity).

```python
check = cp.checks.BandwidthSensitivity(
    bandwidth_start=0.1, bandwidth_stop=1.0, bandwidth_step=0.1
)
result = check.run(experiment)
```

## Test 7: Prior Sensitivity (Bayesian)

**Alternative being tested:** The result is driven by prior choices, not data.

```python
check = cp.checks.PriorSensitivity(
    prior_range={"sigma": [0.5, 1.0, 2.0, 5.0]}
)
result = check.run(experiment)
```

## Test 8: McCrary Density Test (RD)

**Alternative being tested:** Units manipulate their position relative to the threshold (sorting).

```python
check = cp.checks.McCraryDensityTest()
result = check.run(experiment)
# PASS = no density discontinuity (good)
# FAIL = suspicious bunching at threshold (manipulation likely)
```

## Test 9: Convex Hull Check (SC)

**Alternative being tested:** The synthetic control is extrapolating beyond the range of donors.

```python
check = cp.checks.ConvexHullCheck()
result = check.run(experiment)
```

## Test 10: Persistence Check

**Alternative being tested:** The effect is temporary or reverses (suggesting it's an artifact of the intervention window).

```python
check = cp.checks.PersistenceCheck()
result = check.run(experiment)
```

## Running Multiple Tests in a Pipeline

```python
result = cp.Pipeline(
    data=df,
    steps=[
        cp.EstimateEffect(
            method=cp.InterruptedTimeSeries,
            treatment_time=treatment_time,
            formula="y ~ 1 + t",
            model=cp.pymc_models.LinearRegression(
                sample_kwargs={"draws": 1000, "random_seed": 42}
            ),
        ),
        cp.SensitivityAnalysis(
            checks=[
                cp.checks.PreTreatmentPlaceboCheck(),
                cp.checks.PlaceboInTime(n_folds=4),
                cp.checks.OutcomeFalsification(
                    outcome_column="y", falsification_column="z"
                ),
                cp.checks.PriorSensitivity(
                    prior_range={"sigma": [0.5, 1.0, 2.0]}
                ),
            ],
        ),
        cp.SensitivitySummary(),
    ],
).run()

for sr in result.sensitivity_results:
    print(f"{sr.check_name}: {'RULED OUT' if sr.check_passed else 'THREAT REMAINS'}")
```
