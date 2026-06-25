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
      An interrupted time series design with added comparison time series observations. In CITS, one or more control units are included as predictors in the model, providing a counterfactual by accounting for common trends and shocks that affect both treated and control units. This approach provides stronger causal inference than standard ITS by controlling for confounding trends. CITS can be implemented using the ``InterruptedTimeSeries`` class by including control unit observations as predictors in the model formula (e.g., ``treated ~ 1 + control``). Unlike synthetic control, CITS typically includes an intercept and has no sum-to-1 constraint on control weights. Additional covariates such as seasonality, time trends, or other predictors can also be included alongside control units.

   Confidence interval
   CI
      In frequentist statistics, a range of values that would contain the true parameter in a specified percentage of repeated samples. For example, a 95% confidence interval means that if we repeated the study many times, 95% of such intervals would contain the true parameter. See :doc:`reporting_statistics` for interpretation guidance and comparison with credible intervals.

   Confound
      Anything besides the treatment which varies across the treatment and control conditions.

   Convex hull condition
      In synthetic control methods, the requirement that the treated unit's pre-intervention outcomes can be expressed as a convex combination (non-negative weights summing to one) of control units' outcomes. This is a necessary condition for the weighted sum approach to produce a valid synthetic control. When violated—such as when all control series are consistently above or below the treated series—the method cannot construct an accurate counterfactual :footcite:p:`abadie2010synthetic`.

   Counterfactual
      A hypothetical outcome that could or will occur under specific hypothetical circumstances.

   Credible interval
      In Bayesian statistics, an interval containing a specified probability of the posterior distribution. For example, a 95% credible interval contains 95% of the posterior probability mass. Unlike confidence intervals, this is a direct probability statement about the parameter. The HDI (Highest Density Interval) is a specific type of credible interval. See :doc:`reporting_statistics` for details.

   Difference in differences
   DiD
      Analysis where the treatment effect is estimated as a difference between treatment conditions in the differences between pre-treatment to post treatment observations.

   Donor pool
   Donor pool selection
      In synthetic control methods, the donor pool is the set of untreated units available to construct the synthetic control. Donor pool selection (or curation) is the process of choosing which untreated units to include. Units that are structurally dissimilar to the treated unit -- for example, those with negative pre-treatment correlations -- should be excluded because they can introduce interpolation bias and degrade the synthetic control fit. This is especially important in Bayesian implementations where priors (e.g. Dirichlet) assign non-zero weight to every donor by construction :footcite:p:`abadie2021using,abadie2010synthetic`.

   Donut regression discontinuity
   Donut RDD
      A robustness approach for regression discontinuity designs where observations within a specified distance from the treatment threshold are excluded from model fitting. This technique is used when observations closest to the cutoff may be problematic due to manipulation, sorting, or heaping/rounding of the running variable. By excluding the "donut hole" around the threshold, the analysis relies on observations that are less likely to be affected by such issues. See :footcite:t:`noack2024donut` for formal discussion of donut RDD properties.

   Event study
   Event-time effects
      Treatment effect estimates indexed by time relative to treatment onset.

   Interrupted time series design
   ITS
      A quasi-experimental design to estimate a treatment effect where a series of observations are collected before and after a treatment. No control group is present.

   Instrumental Variable regression
   IV
      A quasi-experimental design  to estimate a treatment effect where the is a risk of confounding between the treatment and the outcome due to endogeniety.

   Endogenous Variable
      An endogenous variable is a variable in a regression equation such that the variable is correlated with the error term of the equation i.e. correlated with the outcome variable (in the system). This is a problem for OLS regression estimation techniques because endogeniety violates the assumptions of the Gauss Markov theorem.

   Effect decay
      In interrupted time series analysis with temporary interventions, effect decay refers to the reduction in treatment effect magnitude over time after the intervention ends. Decay patterns can be exponential (rapid initial decline), linear (steady decline), or step (sudden drop to zero). Analyzing decay helps understand how long intervention effects persist and whether they fade gradually or disappear abruptly.

   Effect persistence
      In interrupted time series analysis with temporary interventions, effect persistence refers to the extent to which treatment effects continue after the intervention period ends. It is measured by comparing post-intervention effects to intervention-period effects, often expressed as a percentage (e.g., "30% of the intervention effect persisted"). High persistence suggests lasting behavioral or structural changes, while low persistence indicates temporary effects that fade quickly.

   Intervention period
      In three-period interrupted time series designs, the intervention period is the time window between the treatment start time (``treatment_time``) and treatment end time (``treatment_end_time``) when the intervention is actively applied. This period is distinct from the pre-intervention period (before treatment starts) and the post-intervention period (after treatment ends), enabling separate analysis of immediate effects during treatment versus persistent effects after treatment ends.

   HDI
   Highest Density Interval
      In Bayesian statistics, the narrowest credible interval containing a specified percentage of the posterior probability mass. For example, a 95% HDI is the shortest interval that contains 95% of the posterior distribution. This is the default uncertainty interval reported by CausalPy for PyMC models. See :doc:`reporting_statistics` for interpretation guidance.

   Heaping
   Rounding in running variable
      In regression discontinuity designs, heaping refers to the clustering of running variable values at specific points (typically round numbers) due to rounding or measurement conventions. This can create mass points in the distribution that may be correlated with outcomes or covariates, potentially biasing local fits near the threshold. Donut RDD is one approach to address heaping-induced bias by excluding observations near the threshold. See :footcite:t:`barreca2016heaping` for analysis of heaping-induced bias.

   Local Average Treatment effect
   LATE
      Also known as the complier average causal effect (CACE), is the effect of a treatment for subjects who comply with the experimental treatment assigned to their sample group. It is the quantity we're estimating in IV designs.

   Manipulation
   Sorting at the threshold
      In regression discontinuity designs, manipulation (or sorting) refers to the concern that agents may precisely influence their running variable value to fall just above or below the treatment threshold. For example, students might strategically score just above a passing grade threshold. Such behavior violates the continuity assumptions underlying RDD and can bias treatment effect estimates. The McCrary density test is a common diagnostic for detecting manipulation by testing for discontinuities in the density of the running variable at the threshold. See :footcite:t:`mccrary2008manipulation` for the formal density test.

   Non-equivalent group designs
   NEGD
      A quasi-experimental design where units are assigned to conditions non-randomly, and not according to a running variable (see Regression discontinuity design). This can be problematic when assigning causal influence of the treatment - differences in outcomes between groups could be due to the treatment or due to differences in the group attributes themselves.

   One-group posttest-only design
      A design where a single group is exposed to a treatment and assessed on an outcome measure. There is no pretest measure or comparison group.

   Parallel trends assumption
      An assumption made in difference in differences designs that the trends (over time) in the outcome variable would have been the same between the treatment and control groups in the absence of the treatment.

   Panel data
      Time series data collected on multiple units where the same units are observed at each time point.

   Staggered Difference-in-Differences
      A difference-in-differences setting where different units receive treatment at different times (staggered adoption), often analyzed with event-time treatment effects.

   Absorbing treatment
      A treatment that, once applied, remains in effect permanently. This assumption is required by the Borusyak, Jaravel, and Spiess (2024) imputation estimator used by ``StaggeredDifferenceInDifferences``.

   Posterior probability
      In Bayesian statistics, the probability of a hypothesis or parameter value after observing the data. In CausalPy's reporting, posterior probabilities are used for hypothesis testing (e.g., the probability that a treatment effect is positive). Unlike p-values, these are direct probability statements about the hypothesis of interest. See :doc:`reporting_statistics` for examples.

   Potential outcomes
      A potential outcome is definable for a candidate or experimental unit under a treatment regime with respect to a measured outcome. The outcome Y(0) for that experimental unit is the outcome when the individual does not have the treatment. The outcome Y(1) for that experimental unit is the outcome when the individual does receive the treatment. Only one case can be observed in reality, and this is called the fundamental problem of causal inference. Seen this way causal inference becomes a kind of imputation problem.

   Pretest-posttest design
      A quasi-experimental design where the treatment effect is estimated by comparing an outcome measure before and after treatment.

   Probabilistic Programming
      Probabilistic programming is the practice of expressing statistical using general-purpose programming languages extended with constructs for random variables, probability distributions, and inference. Prominent examples are `PyMC` and `Stan`

   Propensity scores
      An estimate of the probability of adopting a treatment status. Used in re-weighting schemes to balance observational data.

   p-value
      In frequentist statistics, the probability of observing data at least as extreme as what was observed, assuming the null hypothesis (typically "no effect") is true. Lower p-values indicate stronger evidence against the null hypothesis. Commonly, p < 0.05 is used as a threshold for statistical significance, though the p-value itself should be reported along with effect sizes and confidence intervals. See :doc:`reporting_statistics` for interpretation guidance and common pitfalls.

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

   ROPE
   Region of Practical Equivalence
      In Bayesian causal inference, a method for testing whether an effect exceeds a minimum meaningful threshold (the "minimum effect size"). Rather than just testing if an effect differs from zero (which may be statistically significant but trivially small), ROPE analysis tests if the effect is large enough to be practically important. CausalPy reports this as `p_rope`, the posterior probability that the effect exceeds the specified threshold. See :doc:`reporting_statistics` for usage and interpretation.

   Running variable
      In regression discontinuity designs, the running variable is the variable that determines the assignment of units to treatment or control conditions. This is typically a continuous variable. Examples could include a test score, age, income, or spatial location. But the running variable would not be time, which is the case in interrupted time series designs.

   Sharp regression discontinuity design
      A Regression discontinuity design where allocation to treatment or control is determined by a sharp threshold / step function. Common robustness checks for sharp RDD include donut regression discontinuity (excluding observations very close to the threshold) and bandwidth sensitivity analysis. See :footcite:t:`lee2010regression` for a comprehensive overview of RDD methods.

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
