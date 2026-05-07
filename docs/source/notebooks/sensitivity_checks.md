# Sensitivity Checks in Pipeline Workflows

Sensitivity checks help you assess whether a causal estimate is robust or fragile. In quasi-experimental work, they are best treated as design diagnostics that probe assumptions and modelling choices, not as proofs that identification succeeded {cite:p}`reichardt2019quasi,shadish_cook_cambell_2002`.

CausalPy's pipeline API makes sensitivity analysis a first-class step, so robustness checks can run alongside model fitting and report generation in a single, reproducible workflow.

## Architecture overview

The sensitivity framework has three main pieces:

1. **`Check`** --- a protocol that individual checks implement. Each check declares which experiment types it applies to (`applicable_methods`), validates preconditions, and returns a structured `CheckResult`.
2. **`SensitivityAnalysis`** --- a pipeline step that holds a list of `Check` objects and runs them against the fitted experiment.
3. **`CheckResult`** --- the output of a check, containing a pass/fail verdict (or `None` for informational checks), a prose summary, an optional diagnostics table, optional figures, and arbitrary metadata.

When a `GenerateReport` step follows `SensitivityAnalysis`, those results are included in the generated HTML report automatically.

## Choosing checks

### Start with the default suite

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
        cp.SensitivityAnalysis.default_for(cp.InterruptedTimeSeries),
        cp.GenerateReport(),
    ],
).run()
```

`SensitivityAnalysis.default_for(method)` returns a pre-loaded step containing every check currently registered as a default for that experiment type. At present, `PlaceboInTime` is registered as the default check for `InterruptedTimeSeries` and `SyntheticControl`.

### Compose a custom suite

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

`SensitivityAnalysis` checks applicability as it runs. If a check does not support the fitted experiment type, CausalPy raises a clear error naming the methods that check supports.

## Quick reference

| Check | Applies to | Registered as default? | Main question |
|-------|------------|------------------------|---------------|
| {doc}`PlaceboInTime <../api/generated/causalpy.checks.placebo_in_time.PlaceboInTime>` | ITS, SC (PyMC models) | Yes, for ITS and SC | Do pseudo-interventions in the pre-period also produce "effects"? |
| {doc}`PriorSensitivity <../api/generated/causalpy.checks.prior_sensitivity.PriorSensitivity>` | ITS, DiD, SC, Staggered DiD, RD, RKink, PrePostNEGD, IPW, IV (PyMC models) | No | Do conclusions change materially under reasonable prior alternatives? |
| {doc}`PersistenceCheck <../api/generated/causalpy.checks.persistence.PersistenceCheck>` | Three-period ITS designs | No | Does the effect remain after the intervention ends? |
| {doc}`ConvexHullCheck <../api/generated/causalpy.checks.convex_hull.ConvexHullCheck>` | SC | No | Is the treated unit supported by the donor pool, or are we extrapolating? |
| {doc}`LeaveOneOut <../api/generated/causalpy.checks.leave_one_out.LeaveOneOut>` | SC | No | Does the result depend heavily on one donor unit? |
| {doc}`PlaceboInSpace <../api/generated/causalpy.checks.placebo_in_space.PlaceboInSpace>` | SC | No | Are placebo effects in control units as large as the treated effect? |
| {doc}`BandwidthSensitivity <../api/generated/causalpy.checks.bandwidth.BandwidthSensitivity>` | RD, RKink | No | Does the estimate depend heavily on bandwidth choice? |
| {doc}`McCraryDensityTest <../api/generated/causalpy.checks.mccrary.McCraryDensityTest>` | RD | No | Is there evidence of manipulation around the cutoff? |
| {doc}`PreTreatmentPlaceboCheck <../api/generated/causalpy.checks.pre_treatment_placebo.PreTreatmentPlaceboCheck>` | Staggered DiD | No | Do pre-treatment event-study effects look close to zero? |

## Where examples already exist

- `PlaceboInTime`: {doc}`pipeline_workflow`, {doc}`report_demo`
- `BandwidthSensitivity`: {doc}`rkink_pymc`
- `PreTreatmentPlaceboCheck`: {doc}`staggered_did_pymc`
- More check-specific walkthroughs are still being added, so some checks currently have API coverage but no dedicated notebook example yet.

## Check-by-check guide

### {doc}`PlaceboInTime <../api/generated/causalpy.checks.placebo_in_time.PlaceboInTime>`

`PlaceboInTime` moves the intervention backward into the pre-treatment period and re-fits the model. If those pseudo-interventions often produce effects comparable to the observed one, the original result looks less credible. In synthetic control settings, placebo and falsification exercises are a standard part of design assessment {cite:p}`abadie2021using`; in interrupted time series settings, the same logic aligns with broader falsification practice in pre/post intervention designs {cite:p}`lopezbernal2017its`.

This check requires a PyMC-backed model because it works with posterior impact draws. In CausalPy it can also fit a hierarchical null model and, optionally, estimate Bayesian assurance for a user-supplied expected effect prior.

### {doc}`PriorSensitivity <../api/generated/causalpy.checks.prior_sensitivity.PriorSensitivity>`

`PriorSensitivity` re-fits the same experiment with alternative prior specifications and compares the resulting effect summaries. Use it when prior choice could matter materially, especially in small samples or weakly identified models. Reporting how posterior conclusions change under reasonable alternatives is good Bayesian practice {cite:p}`liBayesianProp`.

This is the broadest check in the current API, but it is only available for PyMC-backed experiments.

### {doc}`PersistenceCheck <../api/generated/causalpy.checks.persistence.PersistenceCheck>`

`PersistenceCheck` applies to three-period ITS designs with `treatment_end_time`. It wraps `analyze_persistence()` to ask whether the effect persists, fades, or reverses after the intervention ends. This is especially relevant when policy or campaign effects may decay after treatment is removed {cite:p}`wagner2002segmented`.

### {doc}`ConvexHullCheck <../api/generated/causalpy.checks.convex_hull.ConvexHullCheck>`

`ConvexHullCheck` asks whether the treated unit sits within the support of the donor pool in the pre-treatment period. If not, the synthetic control fit relies on extrapolation rather than interpolation, which weakens design credibility.

### {doc}`LeaveOneOut <../api/generated/causalpy.checks.leave_one_out.LeaveOneOut>`

`LeaveOneOut` drops one control unit at a time and re-fits the synthetic control. If the estimated effect changes dramatically when a single donor is removed, the result depends too heavily on that donor rather than on the donor pool as a whole.

### {doc}`PlaceboInSpace <../api/generated/causalpy.checks.placebo_in_space.PlaceboInSpace>`

`PlaceboInSpace` re-labels each control unit as though it were treated and compares those placebo effects to the observed treated effect. If many placebo units show effects as large as the treated unit, the original estimate looks less distinctive {cite:p}`abadie2010synthetic`.

### {doc}`BandwidthSensitivity <../api/generated/causalpy.checks.bandwidth.BandwidthSensitivity>`

`BandwidthSensitivity` re-fits RD or RKink models across a sequence of bandwidths. Because bandwidth choice drives the bias-variance trade-off in local designs, a result that flips across plausible bandwidths should be treated cautiously {cite:p}`imbens2008regression,lee2010regression`.

### {doc}`McCraryDensityTest <../api/generated/causalpy.checks.mccrary.McCraryDensityTest>`

`McCraryDensityTest` checks for a discontinuity in the density of the running variable at the threshold. A sharp jump suggests units may have manipulated their assignment variable, undermining the design's local comparability assumption {cite:p}`mccrary2008manipulation`.

### {doc}`PreTreatmentPlaceboCheck <../api/generated/causalpy.checks.pre_treatment_placebo.PreTreatmentPlaceboCheck>`

`PreTreatmentPlaceboCheck` examines pre-treatment event-study effects in staggered DiD. If negative event times are far from zero, the parallel trends story is harder to defend and the treatment effect may be biased {cite:p}`goodman2021difference,borusyak2024revisiting`.

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
    status = (
        "PASS"
        if cr.passed is True
        else ("FAIL" if cr.passed is False else "INFO")
    )
    print(f"[{status}] {cr.check_name}: {cr.text}")

    if cr.table is not None:
        display(cr.table)
```

When a `GenerateReport` step follows `SensitivityAnalysis` in the pipeline, check results are automatically included in the HTML report.

## Interpreting sensitivity results

:::{important}
Sensitivity checks are **diagnostics**, not definitive verdicts. A passing check does not prove your causal claim is correct, and a failing check does not prove it is wrong. They reveal where your analysis is robust and where it is fragile.
:::

**What a passing check tells you:** The estimate survived a specific stress test. This increases confidence in the result, especially when multiple independent checks point in the same direction.

**What a failing check tells you:** The estimate is sensitive to a particular assumption or modelling choice. This does not invalidate the analysis; it tells you where to investigate further, justify your choices, or present stronger caveats.

**General guidance:**

- Start with the defaults, then add method-specific checks that target the most plausible failure modes for your design.
- Run more than one check. No single diagnostic is sufficient.
- Report failures as well as passes. Selective reporting of only passing checks undermines credibility.
- Use domain knowledge to decide which failures are consequential. A bandwidth warning at an extreme specification is different from strong placebo evidence.
- Treat checks as part of a cumulative argument, not a mechanical accept/reject gate.

## Next steps

For the pipeline mechanics, see {doc}`pipeline_workflow`. For HTML reporting of check results, see {doc}`report_demo`. More method-specific sensitivity walkthroughs will be added over time; where they already exist, they are linked above.

:::{seealso}
- {doc}`pipeline_workflow` --- end-to-end pipeline tutorial
- {doc}`report_demo` --- HTML report generation
- {doc}`staggered_did_pymc` --- staggered DiD example with `PreTreatmentPlaceboCheck`
- {doc}`rkink_pymc` --- regression kink example with `BandwidthSensitivity`
- {doc}`../knowledgebase/reporting_statistics` --- statistical concepts used in CausalPy reporting
:::

## References

:::{bibliography}
:filter: docname in docnames
:::
