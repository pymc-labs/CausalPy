# Quasi-experimental design notation

This page provides a concise summary of the tabular notation used by {cite:t}`shadish_cook_cambell_2002` and {cite:t}`reichardt2019quasi`. This notation provides a compact description of various experimental designs. While it is possible to describe randomised designs using this notation, we focus purely on {term}`quasi-experimental<Quasi-experiment>` designs here, with non-random allocation (abbreviated as `NR`). Observations are denoted by $O$. Time proceeds from left to right, so observations made through time are labelled as $O_1$, $O_2$, etc. The treatment is denoted by `X`. Rows represent different groups of units. Remember, a unit is a person, place, or thing that is the subject of the study.

## Pretest-posttest designs

One of the simplest designs is the pretest-posttest design. Here we have one row, denoting a single group of units. There is an `X` which means all are treated. The pretest is denoted by $O_1$ and the posttest by $O_2$. See p99 of {cite:t}`reichardt2019quasi`.

|    |   |    |
|----|---|----|
$O_1$ | X | $O_2$ |

Informally, if we think about drawing conclusions about the {term}`causal impact` of the treatment based on the change from $O_1$ to $O_2$, we might say that the treatment caused the change. However, this is a tenuous conclusion because we have no way of knowing what would have happened in the absence of the treatment.

A variation of this design which may (slightly) improve this situation from the perspective of making causal claims, would be to take multiple pretest measures. This is shown below, see p107 of {cite:t}`reichardt2019quasi`.

|    |  |   |    |
|----|--|---|----|
$O_1$ | $O_2$ | X | $O_3$ |

This would allow us to estimate how the group was changing over time before the treatment was introduced. This could be used to make a stronger causal claim about the impact of the treatment.

## Nonequivalent group designs

In randomized experiments, with large enough groups, the randomization process should ensure that the treatment and control groups are approximately equivalent in terms of their attributes. This is positive for causal inference as we can be more sure that differences between control and test groups are due to treatment exposure, not because of differences in attributes of the groups.

However, in quasi-experimental designs, with non-random (`NR`) allocation, we could expect there to be differences between the treatment and control groups' attributes. This poses some challenges in making strong causal claims about the impact of the treatment - we can't be sure that differences between the groups at the posttest are due to the treatment, or due to pre-existing differences between the groups.

In the simplest {term}`nonequivalent group design<NEGD>`, we have two groups, one treated and one not treated, and just one posttest. See p114 of {cite:t}`reichardt2019quasi`.

|     |   |    |
|-----|---|----|
| NR: | X | $O_1$ |
| NR: |   | $O_1$ |

The above design would be considered weak - the lack of a pre-test measure makes it hard to know whether differences between the groups at $O_1$ are due to the treatment or to pre-existing differences between the groups.

This limitation can be addressed by adding a pretest measure. See p115 of {cite:t}`reichardt2019quasi`.

|     |    |   |    |
|-----|----|---|----|
| NR: | $O_1$ | X | $O_2$ |
| NR: | $O_1$ |   | $O_2$ |

Non-equivalent group designs like this, with a pretest and a posttest measure could be analysed in a number of ways:
1. **{term}`ANCOVA`:** Here, the group would be a categorical predictor, the pretest measure would be a covariate, and the posttest measure would be the outcome.
2. **{term}`Difference in differences`:** We can apply linear modeling approaches such as `y ~ group + time + group:time` to estimate the treatment effect. Here, `y` is the outcome measure, `group` is a binary variable indicating treatment or control group, and `time` is a binary variable indicating pretest or posttest. Note that this approach has a strong assumption of [parallel trends](https://en.wikipedia.org/wiki/Difference_in_differences#Assumptions) - that the treatment and control groups would have changed in the same way in the absence of the treatment.

A limitation of the nonequivalent group designs with single pre and posttest measures is that we don't know how the groups were changing over time before the treatment was introduced. This can be addressed by adding multiple pretest measures and can help in assessing if the parallel trends assumption is reasonable. See p154 of {cite:t}`reichardt2019quasi`.

|     |    |   | |    |
|-----|----|---|-|----|
| NR: | $O_1$ | $O_2$ | X | $O_3$ |
| NR: | $O_1$ | $O_2$ |   | $O_3$ |

Again, this design could be analysed using the difference-in-differences approach.

## Interrupted time series designs

While there is no control group, the {term}`interrupted time series design` is a powerful quasi-experimental design that can be used to estimate the causal impact of a treatment. The design involves multiple pretest and posttest measures. The treatment is introduced at a specific point in time, denoted by `X`. The design can be used to estimate the causal impact of the treatment by comparing the trajectory of the outcome variable before and after the treatment. See p203 of {cite:t}`reichardt2019quasi`.

|     |    |   |    |   |    |    |    |    |
|-----|----|---|----|---|----|----|----|----|
| $O_1$ | $O_2$ | $O_3$ | $O_4$ | X | $O_5$ | $O_6$ | $O_7$ | $O_8$ |

## Comparative interrupted time series designs

The {term}`comparative interrupted time-series<CITS>` design incorporates aspects of **interrupted time series** (with only a treatment group), and **nonequivalent group designs** (with a treatment and control group). This design can be used to estimate the causal impact of a treatment by comparing the trajectory of the outcome variable before and after the treatment in the treatment group, and comparing this to the trajectory of the outcome variable in the control group. See p226 of {cite:t}`reichardt2019quasi`.

|     |    |   |    |   |    |    |    |    | |
|-----|----|---|----|---|----|----|----|----|-|
| NR: | $O_1$ | $O_2$ | $O_3$ | $O_4$ | X | $O_5$ | $O_6$ | $O_7$ | $O_8$ |
| NR: | $O_1$ | $O_2$ | $O_3$ | $O_4$ |   | $O_5$ | $O_6$ | $O_7$ | $O_8$ |


Because this design is very similar to the nonequivalent group design, simply with multiple pre and post test measures, it is well-suited to analysis under the difference-in-differences approach.

However, if we have many untreated units and one treated unit, then this design could be analysed with the {term}`synthetic control` approach.

## Regression discontinuity designs

The design notation for {term}`regression discontinuity designs<RDD>` are different from the others and take a bit of getting used to. We have two groups, but allocation to the groups are determined by a units' relation to a cutoff point `C` along a running variable. Also, $O_1$ now represents the value of the running variable, and $O_2$ represents the outcome variable. See p169 of {cite:t}`reichardt2019quasi`. This will make more sense if you consider the design notation alongside one of the example notebooks.

|     |    |   |    |
|-----|----|---|----|
| C: | $O_1$ | X | $O_2$ |
| C: | $O_1$ |   | $O_2$ |

From an analysis perspective, regression discontinuity designs are very similar to interrupted time series designs. The key difference is that treatment is determined by a cutoff point along a running variable, rather than by time.

## Summary
This page has offered a brief overview of the tabular notation used to describe quasi-experimental designs. The notation is a useful tool for summarizing the design of a study, and can be used to help identify the strengths and limitations of a study design. But readers are strongly encouraged to consult the original sources when assessing the relative strengths and limitations of making causal claims under different quasi-experimental designs.

## References
:::{bibliography}
:filter: docname in docnames
:::
