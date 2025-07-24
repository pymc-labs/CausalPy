Glossary
========

.. glossary::
   :sorted:

   ANCOVA
      Analysis of covariance is a simple linear model, typically with one continuous predictor (the covariate) and a catgeorical variable (which may correspond to treatment or control group). In the context of this package, ANCOVA could be useful in pre-post treatment designs, either with or without random assignment. This is similar to the approach of difference in differences, but only applicable with a single pre and post treatment measure.

   Average treatment effect
   ATE
      The average treatment effect across all units.

   Average treatment effect on the treated
   ATT
      The average effect of the treatment on the units that received it. Also called Treatment on the treated.

   Change score analysis
      A statistical procedure where the outcome variable is the difference between the posttest and protest scores.

   Causal impact
      An umbrella term for the estimated effect of a treatment on an outcome.

   Comparative interrupted time-series
   CITS
      An interrupted time series design with added comparison time series observations.

   Confound
      Anything besides the treatment which varies across the treatment and control conditions.

   Counterfactual
      A hypothetical outcome that could or will occur under specific hypothetical circumstances.

   Difference in differences
   DiD
      Analysis where the treatment effect is estimated as a difference between treatment conditions in the differences between pre-treatment to post treatment observations.

   Interrupted time series design
   ITS
      A quasi-experimental design to estimate a treatment effect where a series of observations are collected before and after a treatment. No control group is present.

   Instrumental Variable regression
   IV
      A quasi-experimental design  to estimate a treatment effect where the is a risk of confounding between the treatment and the outcome due to endogeniety.

   Endogenous Variable
      An endogenous variable is a variable in a regression equation such that the variable is correlated with the error term of the equation i.e. correlated with the outcome variable (in the system). This is a problem for OLS regression estimation techniques because endogeniety violates the assumptions of the Gauss Markov theorem.

   Local Average Treatment effect
   LATE
      Also known as the complier average causal effect (CACE), is the effect of a treatment for subjects who comply with the experimental treatment assigned to their sample group. It is the quantity we're estimating in IV designs.

   Non-equivalent group designs
   NEGD
      A quasi-experimental design where units are assigned to conditions non-randomly, and not according to a running variable (see Regression discontinuity design). This can be problematic when assigning causal influence of the treatment - differences in outcomes between groups could be due to the treatment or due to differences in the group attributes themselves.

   One-group posttest-only design
      A design where a single group is exposed to a treatment and assessed on an outcome measure. There is no pretest measure or comparison group.

   Parallel trends assumption
      An assumption made in difference in differences designs that the trends (over time) in the outcome variable would have been the same between the treatment and control groups in the absence of the treatment.

   Panel data
      Time series data collected on multiple units where the same units are observed at each time point.

   Pretest-posttest design
      A quasi-experimental design where the treatment effect is estimated by comparing an outcome measure before and after treatment.

   Propensity scores
      An estimate of the probability of adopting a treatment status. Used in re-weighting schemes to balance observational data.

   Potential outcomes
      A potential outcome is definable for a candidate or experimental unit under a treatment regime with respect to a measured outcome. The outcome Y(0) for that experimental unit is the outcome when the individual does not have the treatment. The outcome Y(1) for that experimental unit is the outcome when the individual does receive the treatment. Only one case can be observed in reality, and this is called the fundamental problem of causal inference. Seen this way causal inference becomes a kind of imputation problem.

   Quasi-experiment
      An empirical comparison used to estimate the effects of a treatment where units are not assigned to conditions at random.

   Random assignment
      Where units are assigned to conditions at random.

   Randomized experiment
      An empirical comparison used to estimate the effects of treatments where units are assigned to treatment conditions randomly.

   Regression discontinuity design
   RDD
      A quasi–experimental comparison to estimate a treatment effect where units are assigned to treatment conditions based on a cut-off score on a quantitative assignment variable (aka running variable).

   Regression kink design
      A quasi-experimental research design that estimates treatment effects by analyzing the impact of a treatment or intervention precisely at a defined threshold or "kink" point in a quantitative assignment variable (running variable). Unlike traditional regression discontinuity designs, regression kink design looks for a change in the slope of an outcome variable at the kink, instead of a discontinuity. This is useful when the assignment variable is not discrete, jumping from 0 to 1 at a threshold. Instead, regression kink designs are appropriate when there is a change in the first derivative of the assignment function at the kink point.

   Running variable
      In regression discontinuity designs, the running variable is the variable that determines the assignment of units to treatment or control conditions. This is typically a continuous variable. Examples could include a test score, age, income, or spatial location. But the running variable would not be time, which is the case in interrupted time series designs.

   Sharp regression discontinuity design
      A Regression discontinuity design where allocation to treatment or control is determined by a sharp threshold / step function.

   Synthetic control
      The synthetic control method is a statistical method used to evaluate the effect of an intervention in comparative case studies. It involves the construction of a weighted combination of groups used as controls, to which the treatment group is compared.

   Treatment on the treated effect
   TOT
      The average effect of the treatment on the units that received it. Also called the average treatment effect on the treated (ATT).

   Treatment effect
      The difference in outcomes between what happened after a treatment is implemented and what would have happened (see Counterfactual) if the treatment had not been implemented, assuming everything else had been the same.

   Wilkinson notation
      A notation for describing statistical models :footcite:p:`wilkinson1973symbolic`.

   Two Stage Least Squares
   2SLS
      An estimation technique for estimating the parameters of an IV regression. It takes its name from the fact that it uses two OLS regressions - a first and second stage.



References
----------
.. footbibliography::
