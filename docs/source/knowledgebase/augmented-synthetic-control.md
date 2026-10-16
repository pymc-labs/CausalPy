---
myst:
  html_meta:
    description: "Bayesian augmented synthetic control: joint simplex weights and a donor prognostic correction, and what the posterior interval does and does not claim."
"og:description": "Bayesian augmented synthetic control: joint simplex weights and a donor prognostic correction, and what the posterior interval does and does not claim."
---

# Augmented synthetic control

This page is the model note for the Bayesian augmented synthetic-control estimator. It records what that estimator conditions on, what `counterfactual` and `treatment_effect` are, and what a posterior interval does and does not claim.

## What the estimator conditions on

The class is `AugmentedSyntheticControl` in `causalpy.experiments.augmented_synthetic_control`. One PyMC model draws the quantities below together. Credible intervals of the counterfactual and the treatment effect are Bayesian intervals for that full model, not a cut posterior.

The panel is a balanced wide outcome frame: time on the index, one shared `treatment_time`, donor columns, and treated columns. The pre-period is every row strictly before `treatment_time`. The post-period starts at `treatment_time`. Both periods must be nonempty.

Weights are independent across treated units. Each unit's weights are Dirichlet with concentration ones, so each draw stays on the simplex. The weight likelihood is Normal, with observation scale 1, on that unit's pre-period outcomes. The mean is the donor pre-period matrix times the unit's weights.

The prognostic regression uses donor outcomes at each time as the response and that donor's full pre-period outcome vector as covariates. Coefficients are time-specific and there is no intercept. They have a Normal(0, `tau`) prior. `tau` is HalfNormal with scale 1. That likelihood is also Normal with observation scale 1. The observation scale is fixed. It is not a prior.

Treated post-period outcomes do not enter either likelihood, the weight update, the prognostic-coefficient update, or the `tau` update. Changing them changes `treatment_effect` only.

## Counterfactual and treatment effect

The untreated counterfactual at post-period time $t$ for treated unit $u$ is

$$
\text{donor\_post}[t] \cdot w_u + (X_u - X_{\text{donors}} w_u) \cdot \beta[t].
$$

$X_u$ and each donor row of $X_{\text{donors}}$ are full pre-period outcome vectors. $\beta[t]$ is the prognostic coefficient vector at time $t$. The first term is the simplex-weighted donor path. The second term is the prognostic correction for the pre-period imbalance.

`treatment_effect` is the treated post-period outcome minus that counterfactual. The treated post-period outcome is a constant in that subtraction. It is not a likelihood contribution and it does not enter `counterfactual`.

The InferenceData names this note uses are `base_weights`, `prognostic_coefficients`, `counterfactual`, `treatment_effect`, and `tau`.

## What a posterior interval does and does not claim

A posterior interval is uncertainty in this one model after integrating the Dirichlet weights, the half-normal scale on the prognostic coefficients, and the two Normal likelihoods. It is not a confidence interval for a two-step estimator, and it does not claim nominal frequentist coverage.

Simplex weights cannot reproduce a treated pre-period that sits outside the convex hull of the donors. The pre-intervention fit then stays imperfect. The prognostic term can still move the post-period counterfactual, because that term is a function of the pre-period imbalance and of coefficients learned from donors only. A shift in the posterior mean on one panel is not evidence that the interval has nominal coverage.

The worked example is [the how-to notebook](../notebooks/augmented-synthetic-control-pymc.ipynb).
