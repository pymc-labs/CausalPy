# Sensitivity Checks in Pipeline Workflows

Sensitivity checks help you assess whether your causal estimate is robust or fragile. A single point estimate from a causal model is rarely enough --- you need to probe it from multiple angles before drawing conclusions.

CausalPy's pipeline API integrates sensitivity analysis as a first-class step, so robustness checks run alongside model fitting and report generation in a single, reproducible workflow.

## Architecture overview

The sensitivity framework has three key components:

1. **`Check`** --- a protocol that individual checks implement. Each check declares which experiment types it applies to (`applicable_methods`), validates preconditions, and returns a structured `CheckResult`.
2. **`SensitivityAnalysis`** --- a pipeline step that holds a list of `Check` objects and runs them against the fitted experiment.
3. **`CheckResult`** --- the output of a check, containing a pass/fail verdict (or `None` for informational checks), a prose summary, an optional diagnostics table, and optional figures.

A typical pipeline looks like this:

```python
import causalpy as cp

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
            checks=[cp.checks.PlaceboInTime(n_folds=3)]
        ),
        cp.GenerateReport(),
    ],
).run()
```

## Choosing checks: default vs custom

### Default checks

`SensitivityAnalysis.default_for(method)` returns a pre-loaded step with all checks that are registered as defaults for a given experiment type:

```python
cp.SensitivityAnalysis.default_for(cp.InterruptedTimeSeries)
```

Currently, `PlaceboInTime` is the only registered default check. This means `default_for()` provides a sensible starting point, but you will typically want to add more checks explicitly.

### Custom check lists

You can compose any combination of checks manually:

```python
cp.SensitivityAnalysis(
    checks=[
        cp.checks.PlaceboInTime(n_folds=4),
        cp.checks.PersistenceCheck(),
        cp.checks.PriorSensitivity(
            alternatives=[
                {"name": "diffuse", "model": cp.pymc_models.LinearRegression(...)},
            ]
        ),
    ]
)
```

The `SensitivityAnalysis` step validates that every check in the list is applicable to the fitted experiment type. If you include a check that doesn't support the experiment, validation fails with a clear error before any check runs.

## Check reference

CausalPy provides nine sensitivity checks, divided into cross-cutting checks that apply broadly and method-specific checks tied to particular experiment types.

### Cross-cutting checks

These checks work across multiple experiment families.

| Check | Applicable methods | What it does |
|-------|-------------------|--------------|
| {py:class}`~causalpy.checks.PlaceboInTime` | ITS, SC | Shifts the treatment time backward into the pre-intervention period and refits the model. If the model finds "effects" where none should exist, the original estimate may be unreliable. Supports a hierarchical null model and optional Bayesian assurance. |
| {py:class}`~causalpy.checks.PriorSensitivity` | All 9 experiment types (Bayesian only) | Re-fits the model with alternative prior specifications and compares posterior estimates. Large sensitivity to priors suggests the data are not informative enough to dominate the prior. |

### Method-specific checks

These checks exploit structure unique to a particular experimental design.

**Interrupted Time Series (ITS)**

| Check | What it does |
|-------|--------------|
| {py:class}`~causalpy.checks.PersistenceCheck` | For three-period ITS designs (with `treatment_end_time`), checks whether the causal effect persists, fades, or reverses after the intervention ends. |

**Synthetic Control (SC)**

| Check | What it does |
|-------|--------------|
| {py:class}`~causalpy.checks.ConvexHullCheck` | Verifies that treated-unit values lie within the convex hull of control units. Violations indicate the synthetic control may be extrapolating beyond the support of the donor pool. |
| {py:class}`~causalpy.checks.LeaveOneOut` | Drops each control unit one at a time, refits the synthetic control, and compares effect estimates. Large variation suggests the result depends heavily on a single donor. |
| {py:class}`~causalpy.checks.PlaceboInSpace` | Treats each control unit as if it were the treated unit (excluding the actual treated unit from the donor pool) and checks whether spurious effects appear. If placebo effects are comparable to the real effect, the finding may not be credible. |

**Regression Discontinuity (RD) and Regression Kink (RKink)**

| Check | What it does |
|-------|--------------|
| {py:class}`~causalpy.checks.BandwidthSensitivity` | Re-fits the model at multiple bandwidths around the threshold and compares effect estimates. A robust effect should be relatively stable across reasonable bandwidth choices. |
| {py:class}`~causalpy.checks.McCraryDensityTest` | Tests for manipulation of the running variable at the threshold by comparing the density of observations just below and just above the cutoff. A discontinuity in density suggests subjects may have manipulated their assignment. |

**Staggered Difference-in-Differences (Staggered DiD)**

| Check | What it does |
|-------|--------------|
| {py:class}`~causalpy.checks.PreTreatmentPlaceboCheck` | Examines pre-treatment event-study estimates. If effects at negative event times are far from zero, the parallel trends assumption is violated and the DiD estimate may be biased. |

## Experiment-to-check matrix

Use this matrix to identify which checks are available for your experiment type.

| Check | ITS | SC | DiD | Staggered DiD | RD | RKink | PrePostNEGD | IPW | IV |
|-------|:---:|:--:|:---:|:-------------:|:--:|:-----:|:-----------:|:---:|:--:|
| PlaceboInTime | ✅ | ✅ | | | | | | | |
| PriorSensitivity | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| PersistenceCheck | ✅ | | | | | | | | |
| ConvexHullCheck | | ✅ | | | | | | | |
| LeaveOneOut | | ✅ | | | | | | | |
| PlaceboInSpace | | ✅ | | | | | | | |
| BandwidthSensitivity | | | | | ✅ | ✅ | | | |
| McCraryDensityTest | | | | | ✅ | | | | |
| PreTreatmentPlaceboCheck | | | | ✅ | | | | | |

## Working with check results

Each check returns a `CheckResult` with the following fields:

- **`passed`** --- `True` if the check passed, `False` if it failed, or `None` for informational checks with no pass/fail criterion.
- **`text`** --- a prose summary describing the outcome.
- **`table`** --- an optional `pandas.DataFrame` with diagnostic statistics.
- **`figures`** --- an optional list of matplotlib figures.
- **`metadata`** --- a dict of arbitrary extra data for downstream steps.

You can inspect results programmatically:

```python
for cr in result.sensitivity_results:
    status = "PASS" if cr.passed else ("FAIL" if cr.passed is False else "INFO")
    print(f"[{status}] {cr.check_name}: {cr.text}")

    if cr.table is not None:
        display(cr.table)
```

When a `GenerateReport` step follows `SensitivityAnalysis` in the pipeline, check results are automatically included in the HTML report.

## Interpreting sensitivity results

:::{important}
Sensitivity checks are **diagnostics**, not definitive verdicts. A passing check does not prove your causal claim is correct, and a failing check does not prove it is wrong. They reveal where your analysis is robust and where it is fragile.
:::

**What a passing check tells you:** The estimate survived a specific stress test. This increases confidence in the result, especially when multiple independent checks pass.

**What a failing check tells you:** The estimate is sensitive to a particular assumption or modelling choice. This does not invalidate the analysis --- it signals where you should investigate further, justify your choices, or present results with appropriate caveats.

**General guidance:**

- Run multiple checks. No single check is sufficient.
- Report all results, including failures. Selective reporting of only passing checks undermines credibility.
- Use domain knowledge to weigh the results. A failing `BandwidthSensitivity` check at an extreme bandwidth may be less concerning than a failing `PlaceboInTime` check.
- Consider the checks as part of your argument, not a mechanical accept/reject gate.

## Next steps

For worked examples showing these checks in action with specific experiment types, see the method-specific walkthroughs in the {doc}`examples index <index>`.

For details on the pipeline API, see {doc}`pipeline_workflow`.

:::{seealso}
- {doc}`pipeline_workflow` --- end-to-end pipeline tutorial
- {doc}`report_demo` --- HTML report generation
- {doc}`../knowledgebase/reporting_statistics` --- statistical concepts used in CausalPy reporting
:::
