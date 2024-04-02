# Experimental design notation

This page provides a concise summary of the tabular notation used by Shadish Cook & Campbell (2002) and Reichardt (2009). This notation provides a compact description of various experimental designs.  While it is possible to describe randomised designs using this notation, we focus purely on quasi-experimental designs here, with non-random allocation (abbreviated as `NR`). Observations are denoted by $O$. Time proceeds from left to right, so observations made through time are labelled as $O_1$, $O_2$, etc. The treatment is denoted by `X`. Rows represent different groups of units. Remember, a unit is a person, place, or thing that is the subject of the study.

## Pretest-posttest designs

One of the simplest designs is the pretest-posttest design. Here we have one row, denoting a single group of units. There is an `X` which means all are treated. The pretest is denoted by $O_1$ and the posttest by $O_2$. See p99 of Reichardt (2019).

|    |   |    |
|----|---|----|
$O_1$ | X | $O_2$ |

Informally, if we think about drawing conclusions about the causal impact of the treatment based on the change from $O_1$ to $O_2$, we might say that the treatment caused the change. However, this is a tenuous conclusion because we have no way of knowing what would have happened in the absence of the treatment.

A variation of this design which may (slightly) improve this situation from the perspective of making causal claims, would be to take multiple pretest measures. This is shown below, see p107 of Reichardt (2019).

|    |  |   |    |
|----|--|---|----|
$O_1$ | $O_2$ | X | $O_3$ |

This would allow us to estimate how the group was changing over time before the treatment was introduced. This could be used to make a stronger causal claim about the impact of the treatment.

## Nonequivalent group designs

In randomized experiments, with large enough groups, the randomization process should ensure that the treatment and control groups are equivalent. However, in quasi-experimental designs, with non-random (`NR`) allocation, we could expect there to be differences between the treatment and control groups. This poses some challenges in making strong causal claims about the impact of the treatment.

For example, in the simplest nonequivalent group design, we have two groups, one treated and one not treated, and just one posttest. See p114 of Reichardt (2019).

|     |   |    |
|-----|---|----|
| NR: | X | $O_1$ |
| NR: |   | $O_1$ |

The above design would be considered weak - the lack of a pre-test measure makes it hard to know whether differences between the groups at $O_1$ are due to the treatment or to pre-existing differences between the groups.

This limitation can be addressed by adding a pretest measure. See p115 of Reichardt (2019).

|     |    |   |    |
|-----|----|---|----|
| NR: | $O_1$ | X | $O_2$ |
| NR: | $O_1$ |   | $O_2$ |

Non-equivalent group designs like this, with a pretest and a posttest measure could be analysed in a number of ways:
1. **ANCOVA:** Here, the group would be a categorical predictor, the pretest measure would be a covariate, and the posttest measure would be the outcome.
2. **Difference-in-differences:** We can apply linear modeling approaches such as `y ~ group + time + group:time` to estimate the treatment effect. Here, `y` is the outcome measure, `group` is a binary variable indicating treatment or control group, and `time` is a binary variable indicating pretest or posttest.

A limitation of the nonequivalent group designs with single pre and posttest measures is that we don't know how the groups were changing over time before the treatment was introduced. This can be addressed by adding multiple pretest measures. See p154 of Reichardt (2019).

|     |    |   | |    |
|-----|----|---|-|----|
| NR: | $O_1$ | $O_2$ | X | $O_3$ |
| NR: | $O_1$ | $O_2$ |   | $O_3$ |

Again, this design could be analysed using the difference-in-differences approach.

## Interrupted time series designs

While there is no control group, the interrupted time series design is a powerful quasi-experimental design that can be used to estimate the causal impact of a treatment. The design involves multiple pretest and posttest measures. The treatment is introduced at a specific point in time, denoted by `X`. The design can be used to estimate the causal impact of the treatment by comparing the trajectory of the outcome variable before and after the treatment. See p203 of Reichardt (2019).

|     |    |   |    |   |    |    |    |    |
|-----|----|---|----|---|----|----|----|----|
| $O_1$ | $O_2$ | $O_3$ | $O_4$ | X | $O_5$ | $O_6$ | $O_7$ | $O_8$ |

## Comparative interrupted time series designs

The comparative interrupted time series design incorporates aspects of **interrupted time series** (with only a treatment group), and **nonequivalent group designs** (with a treatment and control group). This design can be used to estimate the causal impact of a treatment by comparing the trajectory of the outcome variable before and after the treatment in the treatment group, and comparing this to the trajectory of the outcome variable in the control group. See p226 of Reichardt (2019).

|     |    |   |    |   |    |    |    |    | |
|-----|----|---|----|---|----|----|----|----|-|
| NR: | $O_1$ | $O_2$ | $O_3$ | $O_4$ | X | $O_5$ | $O_6$ | $O_7$ | $O_8$ |
| NR: | $O_1$ | $O_2$ | $O_3$ | $O_4$ |   | $O_5$ | $O_6$ | $O_7$ | $O_8$ |


Because this design is very similar to the nonequivalent group design, simply with multiple pre and ppost test measures, it is well-suited to analysis under the difference-in-differences approach.

However, if we have many untreated units and one treated unit, then this design could be analysed with the synthetic control approach.

## Regression discontinuity designs

The design notation for regression discontinuity designs are different from the others and take a bit of getting used to. We have two groups, but allocation to the groups are determined by a units' relation to a cutoff point `C` along a running variable. Also, $O_1$ now represents the value of the running variable, and $O_2$ represents the outcome variable. See p169 of Reichardt (2019). This will make more sense if you consider the design notation alongside one of the example notebooks.

|     |    |   |    |
|-----|----|---|----|
| C: | $O_1$ | X | $O_2$ |
| C: | $O_1$ |   | $O_2$ |

From an analysis perspective, regression discontinuity designs are very similar to interrupted time series designs. The key difference is that treatment is determined by a cutoff point along a running variable, rather than by time.

## Summary
This page has offered a brief overview of the tabular notation used to describe quasi-experimental designs. The notation is a useful tool for summarizing the design of a study, and can be used to help identify the strengths and limitations of a study design. But readers are strongly encouraged to consult the original sources when assessing the relative strengths and limitations of making causal claims under different quasi-experimental designs.
