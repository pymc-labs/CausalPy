# Statistical Reporting in CausalPy

This page explains the statistical concepts used in CausalPy's reporting layer. The reporting functions automatically compute and present statistics appropriate to your model type.

## Model Types and Statistical Approaches

CausalPy supports two modeling frameworks, each with its own statistical paradigm:

| Model Framework | Statistical Approach | Statistics Reported |
|----------------|---------------------|---------------------|
| PyMC models | Bayesian | Mean, Median, HDI, Tail Probabilities, ROPE |
| Scikit-learn models | Frequentist (OLS) | Mean, Confidence Intervals, p-values |

:::{note}
The reporting layer automatically detects which type of model you're using and generates appropriate statistics. You don't need to specify the statistical approach.
:::

---

## Bayesian Statistics (PyMC Models)

When you use PyMC models, CausalPy performs Bayesian inference, yielding posterior distributions for causal effects. The reported statistics summarize these posterior distributions.

### Point Estimates

**Mean**
- The average of the posterior distribution
- Represents the expected value of the causal effect
- **When to use:** Most commonly reported point estimate; balances all posterior information
- **Interpretation:** "The average estimated effect is X"

**Median**
- The middle value of the posterior distribution (50th percentile)
- Divides the posterior probability mass in half
- **When to use:** Preferred when the posterior is skewed; more robust to outliers
- **Interpretation:** "There's a 50% probability the effect is above/below X"

:::{important}
For symmetric posteriors, mean and median are nearly identical. For skewed posteriors, they may differ substantially. Report both to give readers a complete picture.
:::

### Uncertainty Quantification

**HDI (Highest Density Interval)**
- A {term}`credible interval` containing the specified percentage of posterior probability (default: 95%)
- Reported as `hdi_lower` and `hdi_upper` in summary tables
- The narrowest interval containing the specified probability mass
- **Interpretation:** "We can be 95% certain the true effect lies between X and Y"
- **Key difference from CI:** This is a probability statement about the parameter itself, not about the procedure

:::{note}
The `hdi_prob` parameter controls the interval width (e.g., 0.95 for 95% HDI, 0.90 for 90% HDI). Wider intervals (higher probability) provide more certainty but less precision.
:::

**Example interpretation:**
```
mean: 2.5, 95% HDI: [1.2, 3.8]
```
"The estimated effect is 2.5 on average, and we can be 95% certain the true effect lies between 1.2 and 3.8."

### Hypothesis Testing

Bayesian hypothesis testing uses posterior probabilities directly, making the interpretation more intuitive than traditional p-values.

**Directional Tests**
- `p_gt_0`: {term}`Posterior probability` that the effect is greater than zero (positive effect)
- `p_lt_0`: Posterior probability that the effect is less than zero (negative effect)
- **Interpretation:** Direct probability statements about the hypothesis
- **Example:** If `p_gt_0 = 0.95`, there's a 95% probability the effect is positive

**Two-Sided Tests**
- `p_two_sided`: Two-sided posterior probability (analogous to two-sided p-value)
- `prob_of_effect`: Probability of an effect in either direction (1 - p_two_sided)
- **When to use:** When you don't have a directional hypothesis
- **Interpretation:** `prob_of_effect = 0.95` means 95% probability of a non-zero effect

:::{note}
Unlike frequentist p-values, Bayesian posterior probabilities answer the question you actually care about: "What's the probability of this hypothesis given the data?"
:::

**Decision guidance:**
- `p_gt_0 > 0.95` or `p_lt_0 > 0.95`: Strong evidence for directional effect
- `prob_of_effect > 0.95`: Strong evidence for any effect (two-sided)
- Values close to 0.5: Weak or no evidence for the effect

### Effect Size Assessment

**ROPE (Region of Practical Equivalence)**
- Tests whether the effect exceeds a minimum meaningful threshold (`min_effect`)
- Reported as `p_rope` in summary tables
- **Purpose:** Distinguish statistical significance from practical significance
- **Interpretation:** Probability that the effect exceeds the threshold you care about

**How it works:**
1. You specify `min_effect` (the smallest effect size you consider meaningful)
2. For directional tests: `p_rope` = P(|effect| > min_effect)
3. For two-sided tests: `p_rope` = P(|effect| > min_effect)

**Example:**
```python
result.effect_summary(direction="increase", min_effect=1.0)
```
If `p_rope = 0.85`, there's an 85% probability the effect exceeds your meaningful threshold of 1.0.

:::{important}
ROPE analysis requires domain knowledge to set `min_effect`. Consider: What's the smallest effect that would justify the intervention cost? What effect size is scientifically or practically meaningful?
:::

---

## Frequentist Statistics (Scikit-learn / OLS Models)

When you use scikit-learn models (OLS regression), CausalPy performs classical frequentist inference based on t-distributions.

### Point Estimates

**Mean / Coefficient Estimate**
- The estimated causal effect from the regression model
- For scalar effects (DiD, RD): the coefficient of interest
- For time-series effects (ITS, SC): the average or cumulative impact
- **Interpretation:** "The estimated effect is X"

:::{note}
Unlike Bayesian estimates, frequentist point estimates don't come with a probability distribution. Uncertainty is captured through confidence intervals and standard errors.
:::

### Uncertainty Quantification

**Confidence Intervals (CI)**
- Reported as `ci_lower` and `ci_upper` in summary tables
- Computed using t-distribution critical values at the specified significance level (default: α = 0.05 for 95% CI)
- **Interpretation:** "If we repeated this experiment many times, 95% of such intervals would contain the true effect"
- **Key difference from HDI:** This is a statement about the procedure, not about the parameter

**Standard Errors**
- Measure of uncertainty in the coefficient estimate
- Used to construct confidence intervals and compute p-values
- Derived from the residual variance and design matrix
- **Larger standard errors** → wider confidence intervals → more uncertainty

**Example interpretation:**
```
mean: 2.5, 95% CI: [1.1, 3.9]
```
"The estimated effect is 2.5. If we repeated this study many times, 95% of such confidence intervals would contain the true effect."

:::{important}
**Bayesian HDI vs Frequentist CI:** While numerically similar, they have fundamentally different interpretations. The HDI makes a direct probability statement about the parameter ("95% probability the effect is in this range"), while the CI makes a statement about the procedure ("95% of such intervals would contain the true parameter").
:::

### Hypothesis Testing

**p-values**
- The probability of observing data at least as extreme as what we observed, assuming the null hypothesis (no effect) is true
- Reported as `p_value` in summary tables
- **Common threshold:** p < 0.05 is often used as evidence against the null hypothesis
- **Interpretation:** Lower p-values indicate stronger evidence against no effect

**Correct interpretation:**
- p = 0.03: "If there were truly no effect, we'd observe data this extreme only 3% of the time"
- **NOT:** "There's a 97% probability of an effect" (this is a Bayesian interpretation)

**Common pitfalls to avoid:**
1. ❌ "p = 0.06 means no effect" → The p-value doesn't prove the null hypothesis
2. ❌ "p < 0.05 means the effect is important" → Statistical significance ≠ practical significance
3. ❌ "p = 0.01 is better than p = 0.04" → Both provide evidence against the null; the effect size matters more
4. ❌ "p > 0.05 proves no effect" → Absence of evidence is not evidence of absence

**Decision guidance:**
- p < 0.05: Conventional threshold for "statistical significance"
- p < 0.01: Stronger evidence against the null
- p > 0.05: Insufficient evidence to reject the null (but doesn't prove no effect)

:::{note}
Always report the actual p-value and effect size, not just whether p < 0.05. The magnitude and confidence interval of the effect are often more informative than the p-value alone.
:::

**t-statistics and degrees of freedom**
- t-statistic = coefficient / standard error
- Measures how many standard errors the estimate is from zero
- Degrees of freedom (df) = n - p, where n = sample size, p = number of parameters
- Larger |t-statistics| and smaller p-values indicate stronger evidence

---

## Choosing Between Approaches

### When to use Bayesian inference (PyMC models):
- ✅ You want direct probability statements about effects
- ✅ You have prior information to incorporate
- ✅ You need uncertainty quantification for complex hierarchical models
- ✅ You want to test against meaningful effect sizes (ROPE)
- ✅ Small to moderate sample sizes where uncertainty matters

### When to use Frequentist inference (OLS models):
- ✅ You need computational speed (OLS is faster)
- ✅ Your audience expects classical statistical inference
- ✅ Large sample sizes where approaches converge
- ✅ Simple linear models without hierarchy
- ✅ You want to align with traditional econometric practice

:::{important}
Both approaches are valid and will often lead to similar conclusions, especially with larger sample sizes. The choice often depends on your field's conventions, computational constraints, and whether you value direct probabilistic interpretation (Bayesian) or long-run frequency guarantees (frequentist).
:::

---

## Summary Statistics by Effect Type

### Scalar Effects (DiD, RD, Regression Kink)
For experiments with a single causal effect parameter:

**Bayesian output:**
- One row with: mean, median, hdi_lower, hdi_upper
- Tail probabilities: p_gt_0 (or p_lt_0, or p_two_sided + prob_of_effect)
- Optional: p_rope (if min_effect specified)

**Frequentist output:**
- One row with: mean, ci_lower, ci_upper, p_value

### Time-Series Effects (ITS, Synthetic Control)
For experiments with multiple post-treatment time points:

**Two aggregation levels:**
1. **Average effect:** Mean effect across the post-treatment window
2. **Cumulative effect:** Sum of effects across the post-treatment window

**Additional statistics:**
- **Relative effects:** Percentage change relative to counterfactual
  - `relative_mean`: Effect size as percentage of counterfactual
  - `relative_hdi_lower` / `relative_hdi_upper` (Bayesian)
  - `relative_ci_lower` / `relative_ci_upper` (frequentist)

---

## Usage Examples

### Basic usage (default Bayesian):
```python
import causalpy as cp

# Fit experiment with PyMC model
result = cp.DifferenceInDifferences(...)

# Get effect summary with default settings
summary = result.effect_summary()
print(summary.text)  # Prose interpretation
print(summary.table)  # Numerical summary
```

### With directional hypothesis:
```python
# Test for an increase
summary = result.effect_summary(direction="increase")  # Reports p_gt_0

# Test for a decrease
summary = result.effect_summary(direction="decrease")  # Reports p_lt_0

# Two-sided test
summary = result.effect_summary(direction="two-sided")  # Reports prob_of_effect
```

### With practical significance threshold:
```python
# Only care about effects > 2.0
summary = result.effect_summary(
    direction="increase",
    min_effect=2.0  # ROPE analysis
)
# Now summary.table includes p_rope column
```

### For time-series experiments with custom window:
```python
# ITS or Synthetic Control result
summary = result.effect_summary(
    window=(10, 20),  # Only analyze time points 10-20
    cumulative=True,   # Include cumulative effects
    relative=True      # Include percentage changes
)
```

---

## Further Reading

For deeper understanding of these statistical concepts:

- **Bayesian inference:** The [PyMC documentation](https://www.pymc.io/) provides excellent tutorials on Bayesian statistics
- **Causal inference:** See our :doc:`causal_written_resources` for recommended books
- **Statistical terms:** Refer to the :doc:`glossary` for concise definitions
- **Practical application:** Explore the example notebooks in our documentation showing these concepts in action

:::{seealso}
- :doc:`glossary` - Quick reference for statistical terms
- :doc:`causal_written_resources` - Books and articles on causal inference
- API documentation for the `effect_summary()` method
:::
