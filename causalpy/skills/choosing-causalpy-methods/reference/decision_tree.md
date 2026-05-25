# Decision Tree

Use this file as the canonical routing algorithm for choosing a CausalPy experiment. It is written as ordered text for agents, not as a visual tree. Follow the steps in order and stop as soon as the route is matched, ambiguous, not identifiable yet, or not implemented in CausalPy.

## Routing Algorithm

```text
1. Check required intake.
   If estimand, assignment mechanism, data topology, and control type are not known, ask for the single missing fact that most changes the route.

2. Check for unsupported goals before force-fitting.
   If the user needs matching, mediation, survival models, causal forests/CATE, fuzzy RD, non-absorbing staggered treatment, augmented synthetic control, matrix completion, alternative staggered DiD estimators, or continuous/multi-arm treatment outside IV or kink settings, return the Not implemented in CausalPy output.

3. If assignment is a known intervention time, route by data topology.
   If data is one outcome series with no donor units, compare InterruptedTimeSeries and PiecewiseITS.
   If data is one treated outcome series plus one or more comparison/control series used as formula predictors, route to InterruptedTimeSeries as Comparative Interrupted Time Series (CITS), then compare against SyntheticControl if the user wants constrained donor weighting.
   If data is a wide panel where columns are units and at least one untreated donor unit is available, compare SyntheticControl and SyntheticDifferenceInDifferences.
   If data is a long panel with treated and control groups, compare DifferenceInDifferences, StaggeredDifferenceInDifferences, and PanelRegression.

4. If assignment is a common treated/control pre/post design, route to DifferenceInDifferences.
   Use this when treated and control groups share one intervention timing and the estimand is the group-by-post interaction. If adoption timing differs by unit, switch to StaggeredDifferenceInDifferences. If there is only one baseline and one post outcome per unit, compare PrePostNEGD.

5. If assignment is staggered adoption, route to StaggeredDifferenceInDifferences only when treatment is absorbing.
   Use this for cohort or event-time ATT paths with unit-time panel data, no anticipation, and never-treated or not-yet-treated comparison information. If treatment can turn off or repeat, return Not implemented in CausalPy.

6. If assignment is a cutoff or kink in a running variable, route by estimand.
   Use RegressionDiscontinuity for a sharp level jump at a cutoff. Use RegressionKink for a slope or treatment-intensity change at a kink point. If treatment probability changes but treatment is not sharp at the cutoff, return Not implemented in CausalPy unless the problem can be reframed as a general InstrumentalVariable analysis with clear caveats.

7. If assignment depends on a credible instrument, route to InstrumentalVariable.
   Require a relevance story, exclusion restriction, and outcome/treatment/instrument data. If the user mainly has measured confounders rather than an instrument, compare InversePropensityWeighting and PanelRegression instead.

8. If treatment is observed and confounded, route by treatment type and adjustment story.
   Use InversePropensityWeighting only for binary treatment with measured confounders and plausible overlap. Use PanelRegression only when the analysis target is fixed-effects coefficient estimation and the user understands the result is not automatically a causal design. If confounding is unmeasured and there is no valid instrument, return Not identifiable yet or Not implemented in CausalPy rather than forcing IPW.

9. If data contains pre and post outcomes but not a repeated time panel, compare PrePostNEGD and DifferenceInDifferences.
   Use PrePostNEGD for baseline-adjusted nonequivalent group designs with one pretest and one posttest outcome. Use DifferenceInDifferences only when there are repeated observations over time that support a pre/post group trend comparison.

10. Apply capability gates before finalizing.
    If the user requires OLS/sklearn, avoid Bayesian-only methods. If the user requires `effect_summary()` or a unified `plot()`, check the method capability matrix before selecting IV, IPW, or PanelRegression.

11. Return one output contract.
    Return Matched, Ambiguous, Not identifiable yet, or Not implemented in CausalPy. Do not write analysis code as part of this skill unless the route is Matched and the user explicitly asks to proceed.
```

## Leaf Routes

### `InterruptedTimeSeries`

Use when there is one outcome series, a known intervention time, and the counterfactual should be forecast from the pre-intervention trend. This also covers Comparative Interrupted Time Series (CITS) when comparison/control series are included as predictors in the formula. Do not use it when the user wants synthetic-control donor weights or has a treated/control panel DiD design.

### `PiecewiseITS`

Use when there is one outcome series and the estimand is an explicit level and/or slope change at one or more known interruption times, encoded through `step()` or `ramp()` formula terms. Do not use it for unknown changepoint discovery.

### `DifferenceInDifferences`

Use for treated and control groups observed before and after a common intervention timing, where the estimand is the treated-group by post-period contrast. Do not use it for staggered adoption or one-row pre/post snapshots without repeated time structure.

### `StaggeredDifferenceInDifferences`

Use for long unit-time panels where units adopt treatment at different times and treatment is absorbing. Do not use it for reversible, intermittent, or non-absorbing treatments.

### `SyntheticControl`

Use for wide panels with treated unit columns and untreated donor unit columns, where donor weights should reproduce the treated unit's pre-period path. Do not use it when there is no credible donor support.

### `SyntheticDifferenceInDifferences`

Use for the same wide donor-panel setting as synthetic control when the design explicitly relies on both donor-unit weights and pre-period time weights. Treat it as Bayesian-only in practice.

### `PanelRegression`

Use when fixed-effects regression is the target, especially for coefficient-level adjustment with unit and/or time effects. Do not present it as a causal design unless the treatment assignment story justifies that interpretation.

### `PrePostNEGD`

Use for pretest/posttest nonequivalent group designs with a baseline outcome, post outcome, and group indicator. Do not use it for repeated panel time series where DiD assumptions can be assessed.

### `RegressionDiscontinuity`

Use for sharp regression discontinuity designs where treatment jumps at a known cutoff in a running variable. Do not use it for fuzzy RD unless the user accepts a different IV framing.

### `RegressionKink`

Use when treatment intensity or incentives change slope at a known kink point rather than jumping in level. Do not use it for ordinary threshold discontinuities.

### `InstrumentalVariable`

Use when treatment is endogenous and there is a credible instrument. Do not use it just because a covariate predicts treatment; the exclusion restriction must be part of the design story.

### `InversePropensityWeighting`

Use for observational binary treatment with measured confounders and plausible overlap. Do not use it for continuous, multi-arm, or unmeasured-confounding problems.
