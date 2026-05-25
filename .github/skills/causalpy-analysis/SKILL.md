---
name: causalpy-analysis
description: End-to-end causal inference with CausalPy — method selection, data loading, model fitting, results, sensitivity checks, and pipelines. Use when performing any causal analysis task.
---

# CausalPy Analysis

Complete workflow for causal inference using CausalPy. This skill covers the full process from choosing a method to validating results.

## Process

### 1. Select the Right Method

Use the decision framework to choose among 9 experiment types based on your data structure and research question.

**Quick decision tree:**

- Have a control group? → DiD or Synthetic Control
- Single unit, time series? → ITS or PiecewiseITS
- Running variable with threshold? → RD or RK
- Need to handle selection bias? → IPW or IV
- Pre/post with non-equivalent groups? → PrePostNEGD

See [Method Selection](reference/method_selection.md) for the full decision framework and all 9 methods.

### 2. Prepare Data

Prepare a `pd.DataFrame` matching the method's requirements. Each method reference below documents the expected data structure.

For demos and testing, CausalPy ships with built-in example datasets — see the `example-datasets` skill.

### 3. Choose a Model

Every experiment accepts either a **Bayesian** (PyMC) or **OLS** (scikit-learn) model. Not all experiments support both — check the method reference.

| Model | Use Case | Import |
|---|---|---|
| `LinearRegression` | General regression (DiD, ITS, RD, RK) | `cp.pymc_models.LinearRegression` |
| `WeightedSumFitter` | Synthetic Control | `cp.pymc_models.WeightedSumFitter` |
| `PropensityScore` | IPW | `cp.pymc_models.PropensityScore` |
| `InstrumentalVariableRegression` | IV | `cp.pymc_models.InstrumentalVariableRegression` |
| Any sklearn `RegressorMixin` | OLS experiments | `cp.create_causalpy_compatible_class(estimator)` |

See [Models](reference/models.md) for full details and prior specifications.

### 4. Fit the Experiment

Each experiment is initialized with data, model, and method-specific parameters. Fitting happens on construction.

```python
result = cp.DifferenceInDifferences(
    df,
    formula="y ~ 1 + group*post_treatment",
    time_variable_name="t",
    group_variable_name="group",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"draws": 1000, "random_seed": 42}
    ),
)
```

See method-specific references for parameters and examples:
- [Difference-in-Differences](reference/diff_in_diff.md)
- [Interrupted Time Series](reference/interrupted_time_series.md)
- [Synthetic Control](reference/synthetic_control.md)
- [Regression Discontinuity & Kink](reference/regression_discontinuity.md)
- [IPW, IV & PrePostNEGD](reference/propensity_and_iv.md)

### 5. Analyze Results

All experiments share a common results interface:

```python
result.print_coefficients()         # Model coefficients
result.plot()                       # Observed vs counterfactual
result.effect_summary()             # Structured effect summary (table + prose)
result.generate_report()            # Self-contained HTML report
result.idata                        # ArviZ InferenceData (Bayesian only)
```

`effect_summary()` parameters: `window`, `direction`, `alpha`, `cumulative`, `relative`, `min_effect`, `treated_unit`, `period`, `prefix`.

### 6. Run Sensitivity & Diagnostic Checks

CausalPy provides 11 checks across two categories. Run them standalone or in a pipeline.

**Sensitivity checks** (robustness):

- `PlaceboInTime` — hierarchical null model + optional Bayesian assurance
- `PlaceboInSpace` — permutation across units
- `BandwidthSensitivity` — RD/RK bandwidth sensitivity
- `LeaveOneOut` — jackknife
- `PriorSensitivity` — Bayesian prior sensitivity

**Diagnostic checks** (assumptions):

- `PreTreatmentPlaceboCheck` — pre-treatment effects
- `OutcomeFalsification` — falsification on alternative outcomes
- `PersistenceCheck` — effect persistence over time
- `McCraryDensityTest` — density at threshold (RD)
- `ConvexHullCheck` — donor convex hull (SC)

```python
check = cp.checks.PlaceboInTime(n_folds=4)
placebo_result = check.run(experiment=result)
print(placebo_result.text)
```

See [Sensitivity Checks](reference/sensitivity_checks.md) for all checks with parameters and examples.

### 7. Use Pipelines (Optional)

Chain steps into a reproducible pipeline:

```python
result = cp.Pipeline(
    data=df,
    steps=[
        cp.EstimateEffect(
            method=cp.InterruptedTimeSeries,
            treatment_time=pd.Timestamp("2020-03-01"),
            formula="y ~ 1 + t",
            model=cp.pymc_models.LinearRegression(),
        ),
        cp.SensitivityAnalysis(
            checks=[cp.checks.PlaceboInTime(n_folds=4)],
        ),
        cp.SensitivitySummary(),
        cp.GenerateReport(output_file="report.html"),
    ],
).run()
```

See [Pipeline](reference/pipeline.md) for all steps and composition patterns.

## References

| Reference | Contents |
|---|---|
| [Method Selection](reference/method_selection.md) | Decision framework, all 9 methods with support matrix |
| [Models](reference/models.md) | PyMC and sklearn model classes, priors, configuration |
| [Diff-in-Diff](reference/diff_in_diff.md) | DiD and Staggered DiD |
| [Interrupted Time Series](reference/interrupted_time_series.md) | ITS and PiecewiseITS |
| [Synthetic Control](reference/synthetic_control.md) | SC with donor validation |
| [Regression Discontinuity](reference/regression_discontinuity.md) | RD and RK |
| [IPW, IV & PrePostNEGD](reference/propensity_and_iv.md) | Propensity, instruments, pre/post |
| [Sensitivity Checks](reference/sensitivity_checks.md) | All 11 checks with examples |
| [Pipeline](reference/pipeline.md) | Pipeline steps and composition |
