# Sensitivity & Diagnostic Checks

All checks inherit from `Check` protocol and return `CheckResult` objects. Run standalone via `check.run(experiment)` or in a pipeline via `SensitivityAnalysis`.

## Sensitivity Checks (Robustness)

### PlaceboInTime

Shifts treatment time backward to create placebo folds, fits a hierarchical null model, compares actual effect against learned null. Optional Bayesian assurance for power analysis.

```python
cp.checks.PlaceboInTime(
    n_folds=3,                       # number of placebo folds
    experiment_factory=None,         # (data, treatment_time) -> experiment
    sample_kwargs=None,              # MCMC settings for hierarchical model
    threshold=0.95,                  # P(outside null) cutoff
    prior_scale=1.0,                 # prior width multiplier
    expected_effect_prior=None,      # distribution for assurance
    rope_half_width=None,            # ROPE half-width (required with prior)
    n_design_replications=None,      # simulation reps for assurance
    random_seed=None,
)
```

**Result metadata:** `fold_results`, `status_quo_idata`, `null_samples`, `actual_cumulative_mean`, `p_effect_outside_null`, `assurance_result`.

```python
check = cp.checks.PlaceboInTime(n_folds=4)
result = check.run(experiment)
print(result.text)
print(result.metadata["p_effect_outside_null"])
```

**With Bayesian assurance:**

```python
import preliz as pz

check = cp.checks.PlaceboInTime(
    n_folds=4,
    expected_effect_prior=pz.maxent(pz.Normal(), lower=60, upper=120, mass=0.95),
    rope_half_width=50.0,
    random_seed=42,
)
result = check.run(experiment)
ar = result.metadata["assurance_result"]
print(f"Assurance: {ar.true_positive_rate:.1%}")
```

### PlaceboInSpace

Permutation test across units — applies treatment to untreated units to build a null distribution.

```python
cp.checks.PlaceboInSpace(n_placebo_units, n_leads)
```

### BandwidthSensitivity

Tests RD/RK sensitivity to bandwidth choice.

```python
cp.checks.BandwidthSensitivity(
    bandwidth_start, bandwidth_stop, bandwidth_step
)
```

### LeaveOneOut

Jackknife sensitivity — refits excluding one observation at a time.

```python
cp.checks.LeaveOneOut()
```

### PriorSensitivity

Tests Bayesian model sensitivity to prior specifications.

```python
cp.checks.PriorSensitivity(prior_range={"beta": [...], "sigma": [...]})
```

## Diagnostic Checks (Assumptions)

### PreTreatmentPlaceboCheck

Checks for pre-treatment effects that would violate assumptions.

```python
cp.checks.PreTreatmentPlaceboCheck()
```

### OutcomeFalsification

Tests effect on an outcome that should NOT be affected by treatment.

```python
cp.checks.OutcomeFalsification(
    outcome_column="y",                # actual outcome
    falsification_column="placebo_y",  # outcome that should be unaffected
)
```

### PersistenceCheck

Checks whether treatment effects persist over time.

```python
cp.checks.PersistenceCheck()
```

### McCraryDensityTest

Tests for density discontinuity at the RD threshold (manipulation check).

```python
cp.checks.McCraryDensityTest()
```

### ConvexHullCheck

Verifies treated unit lies within the convex hull of donor units (SC validity).

```python
cp.checks.ConvexHullCheck()
```

## Using Checks in a Pipeline

```python
result = cp.Pipeline(
    data=df,
    steps=[
        cp.EstimateEffect(
            method=cp.InterruptedTimeSeries,
            treatment_time=treatment_time,
            formula="y ~ 1 + t",
            model=cp.pymc_models.LinearRegression(),
        ),
        cp.SensitivityAnalysis(
            checks=[
                cp.checks.PlaceboInTime(n_folds=4),
                cp.checks.PriorSensitivity(prior_range={"sigma": [0.5, 1.0, 2.0]}),
                cp.checks.OutcomeFalsification(
                    outcome_column="y", falsification_column="z"
                ),
            ],
        ),
        cp.SensitivitySummary(),
    ],
).run()

for sr in result.sensitivity_results:
    print(sr.check_name, sr.check_passed)
```
