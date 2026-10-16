---
myst:
  html_meta:
    description: "Bayesian interactive fixed-effects synthetic control: Xu's untreated path, shrinkage over rank, and what the posterior interval does and does not claim."
"og:description": "Bayesian interactive fixed-effects synthetic control: Xu's untreated path, shrinkage over rank, and what the posterior interval does and does not claim."
---

# Interactive fixed-effects synthetic control

This page is the model note for the Bayesian interactive fixed-effects estimator. It records the generative process, the shrinkage choice, the identification limits, the estimand, and the panel shapes v1 supports.

## What is being estimated

The estimand is the untreated trajectory of each treated unit: the outcome that unit would have had at each time if it had remained untreated. The impact is the observed treated outcome minus that trajectory. It is a unit-level, time-specific contrast, not an average treatment effect for a broader population, and not a selected-rank point estimate.

Xu (2017) targets the same untreated path with an interactive fixed-effects model and a cross-validated factor rank. This estimator targets that path. It does not reproduce his estimator, his bootstrap, or his numerical values. A posterior interval here is uncertainty in one Bayesian model after integrating the shrinkage prior. It is not a confidence interval for a rank chosen by cross-validation, and it does not claim nominal frequentist coverage.

## Generative model

On permissible cells the mean is

$$
\mu_{it} = \alpha_i + \xi_t + \lambda_i^\top f_t.
$$

$\alpha_i$ is a unit intercept, $\xi_t$ is a time intercept, $\lambda_i$ is the unit's loading, and $f_t$ is the shared factor at time $t$. There are no covariates in v1.

Observations are independent Gaussian noise around that mean, with one shared positive residual scale. That is the noise around the linear factor mean. It is not a claim that serial correlation or heteroscedasticity across time has been modelled. Heavier tails stay out of v1.

Permissible cells are control units at every time, and each treated unit strictly before the shared treatment time. Treated post-intervention outcomes never enter the likelihood, the factor update, the loading update, the intercept update, the shrinkage update, or the residual-scale update. Changing them changes impacts only.

Shared factors and shared time intercepts are informed by the control panel only. The control likelihood uses the one shared positive residual scale, not a unit observation scale. That is one random variable. Treated pre-period outcomes may update that scale's draws after the factor and time-intercept sample, and must not resample those draws. Loadings use the updated draws. The factor posterior is the control-panel posterior under that scale as informed by the control panel. It is not recomputed under the treated-pre update. There is one residual scale, not two, and the caller does not supply it.

## Shrinkage, not a selected rank

v1 fits one factor model with a fixed upper bound `max_factors` and a multiplicative gamma process on the factor scales (Bhattacharya and Dunson). There is one positive scale per factor. Later factors are shrunk at least as hard, in the prior, as earlier ones, and unused scales can go to zero. Posterior intervals for the untreated trajectory marginalize that prior. They are not conditional on a chosen rank.

`max_factors` is configuration, not a selected rank. The result field `shrinkage_state` stores posterior draws of those scales. It is not an integer rank and not an effective-rank count. There is no `n_factors` argument and no cross-validation selector.

The multiplicative-gamma concentrations, the residual-scale prior, and the numeric default of `max_factors` are implementation constants. Replacing the gamma process with another ordered increasing-shrinkage prior would still have to expose one positive scale per factor, still have to be able to drive unused scales toward zero, and still have to be marginalized in the reported intervals.

## What is not identified

The untreated counterfactual is the sum above. Raw factors and raw loadings are not unique: a rotation of one against the other leaves the product unchanged, and a sign flip does the same. v1 does not put orthonormality constraints in the sampler. Do not read raw factors or raw loadings as stable identifiers. Plots should show reconstructed trajectories, not the latent factor series as if those series had a unique meaning.

## Supported panels

v1 uses the balanced wide panel already stored by `WidePanel`: one shared treatment time, one or more treated units, and control units as donors. Missing values in the used columns, staggered treatment times, and covariates are out of bounds.

The class is `InteractiveFixedEffectsSyntheticControl` in `causalpy.experiments.interactive_fixed_effects`. It holds a `WidePanel`. It is not a `SyntheticControl` subclass. The untreated path is the loading-factor product plus the two additive intercepts. It is not a prediction from post-period donor rows. The worked example is [the how-to notebook](../notebooks/interactive-fixed-effects-pymc.ipynb). The cross-model comparison is [Comparing Bayesian synthetic-control models](comparing-bayesian-synthetic-control.md).
