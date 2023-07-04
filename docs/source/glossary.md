# Glossary

:::{note}
Some of the definitions have been copied from (or inspired by) various resources, including {cite:t}`reichardt2019quasi`.
:::

**ANCOVA:** Analysis of covariance is a simple linear model, typically with one continuous predictor (the covariate) and a catgeorical variable (which may correspond to treatment or control group). In the context of this package, ANCOVA could be useful in pre-post treatment designs, either with or without random assignment. This is similar to the approach of difference in differences, but only applicable with a single pre and post treatment measure.

**Average treatment effect (ATE):** The average treatement effect across all units.

**Average treatment effect on the treated (ATT):** The average effect of the treatment on the units that recieved it. Also called Treatment on the treated.

**Change score analysis:** A statistical procedure where the outcome variable is the difference between the posttest and protest scores.

**Comparative interrupted time-series (CITS) design:** An interrupted time series design with added comparison time series observations.

**Confound** Anything besides the treatment which varies across the treatment and control conditions.

**Counterfactual:** A hypothetical outcome that could or will occur under specific hypothetical circumstances.

**Difference in Differences (DID) analysis:** Analysis where the treatment effect is estimated as a difference between treatment conditions in the differences between pre-treatment to post treatment observations.

**Interrupted time series (ITS) design:** A quasi-experimental design to estimate a treatment effect where a series of observations are collected before and after a treatment. No control group is present.

**Instrumental Variable Regression (IVR) design** A quasi experimental design to estimate the treatment effect when there is
a concern that the treatment variable is endogenous in the system of interest.

**Non-equivalent group designs:** A quasi-experimental design where units are assigned to conditions non-randomly, and not according to a running variable (see Regression discontinuity design).

The basic posttest-only design can be described as:

| NR: | X | O |
|-----|---|---|
| NR: |   | 0 |

The pretest-posttest NEGD can be described as:

| NR: | O1 | X | O2 |
|-----|----|---|----|
| NR: | O1 |   | O2 |

Data from pretest-posttest NEGD could be analysed using change score analysis or the difference in differences approach, or with ANCOVA.

**One-group posttest-only design:** A design where a single group is exposed to a treatment and assessed on an outcome measure. There is no pretest measure or comparison group.

**Panel data:** Time series data collected on multiple units where the same units are observed at each time point.

**Pretest-posttest design:** A quasi-experimental design where the treatment effect is estimated by comparing an outcome measure before and after treatment.

**Quasi-experiment:** An empirical comparison used to estimate the effects of a treatment where units are not assigned to conditions at random.

**Random assignment:** Where units are assigned to conditions at random.

**Randomized experiment:** An emprical comparison used to estimate the effects of treatments where units are assigned to treatment conditions randomly.

**Regression discontinuity design:** A quasi–experimental comparison to estimate a treatment effect where units are assigned to treatment conditions based on a cut-off score on a quantitative assignment variable (aka running variable).

**Sharp regression discontinuity design:** A Regression discontinuity design where allocation to treatment or control is determined by a sharp threshold / step function.

**Synthetic control method:** The synthetic control method is a statistical method used to evaluate the effect of an intervention in comparative case studies. It involves the construction of a weighted combination of groups used as controls, to which the treatment group is compared.

**Treatment on the treated (TOT) effect:** The average effect of the treatment on the units that recieved it. Also called the average treatment effect on the treated (ATT).

**Treatment effect:** The difference in outcomes between what happened after a treatment is implemented and what would have happened (see Counterfactual) if the treatment had not been implemented, assuming everything else had been the same.

## References
:::{bibliography}
:filter: docname in docnames
:::
