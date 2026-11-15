# Synthetic-control qualification report (issue-914-qualification-v1)

This is not a launch recommendation. The selected-duration audit is unmet.

gap: no public duration-freeze API. Inspected origin/main at c4fa86726b040ba7164b9396eb5849c90d271b37 and open pull requests. Issue #721 has no open pull request. SyntheticControlPlanner has no freeze method. This delivery does not invent a private duration selector. The selected-duration audit is unmet.

No numeric null-error tolerance is claimed. The 94% HDI is an interval convention, and the 0.95 tail probability is the decision rule. Neither is a false-positive tolerance.

Fold standard deviations are posterior standard deviations of the cumulative impact, not residual observation standard deviations and not future observation noise. The effect posterior subtracts counterfactual conditional-mean draws from observed outcomes. Crossing each learned-null draw with every fold standard deviation treats bias and precision as independent. A curve fit on calibration panels does not by itself transport to audit panels or to a future experiment. Numerical exactness of the engine is not qualification.

Panels requested: 100.
Translation identity matches: True. When it matches, the bias of the cumulative estimand is the same at every lift within a scenario, so those bias rows are not independent estimates.
Launch recommendation qualified: False.
Supported means the protocol conditions for stating the observed count were met. It does not mean the detection target was achieved, and a plug-in standard error below 0.04 is not a qualified precision claim.

## Scenario counts

| Scenario | Lift | Detected | Wrong sign | Null negative tail | Failed | Truth covered | Bias mean | Bias covers zero | Plug-in SE | SE qualified | Operating | Recovery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correctly_specified_noisy | 0.00 | 38 | 0 | 34 | 0 | 33/100 | 0.208 | True | 0.0485 | True | supported | descriptive |
| correctly_specified_noisy | 0.25 | 60 | 14 | 0 | 0 | 33/100 | 0.208 | True | 0.0490 | True | supported | descriptive |
| correctly_specified_noisy | 1.00 | 97 | 1 | 0 | 0 | 33/100 | 0.208 | True | 0.0171 | False | supported | descriptive |
| persistent_level_trend_misfit | 0.00 | 100 | 0 | 0 | 0 | 0/100 | 66.668 | False | 0.0000 | False | descriptive | descriptive |
| persistent_level_trend_misfit | 0.25 | 100 | 0 | 0 | 0 | 0/100 | 66.668 | False | 0.0000 | False | descriptive | descriptive |
| persistent_level_trend_misfit | 1.00 | 100 | 0 | 0 | 0 | 0/100 | 66.668 | False | 0.0000 | False | descriptive | descriptive |
| serially_correlated_errors | 0.00 | 37 | 0 | 35 | 0 | 31/100 | 0.152 | True | 0.0483 | True | supported | descriptive |
| serially_correlated_errors | 0.25 | 50 | 25 | 0 | 0 | 31/100 | 0.152 | True | 0.0500 | True | supported | descriptive |
| serially_correlated_errors | 1.00 | 85 | 10 | 0 | 0 | 31/100 | 0.152 | True | 0.0357 | False | supported | descriptive |

## Hull warnings

correctly_specified_noisy: 95 of 100 panels warned that a pre-period point was outside the donor range. Panels were not dropped.
persistent_level_trend_misfit: 100 of 100 panels warned that a pre-period point was outside the donor range. Panels were not dropped.
serially_correlated_errors: 97 of 100 panels warned that a pre-period point was outside the donor range. Panels were not dropped.

The reference scenario recorded 95 of 100 convex-hull warnings. That count confounds the label, so it is not a correctly specified recovery diagnostic. The amplitude was not changed after the counts were seen.

## Comparison

Curve probabilities use cumulative effects 0, 2, and 8. Audit rates are the audit half, panels 50 through 99, at per-period lifts 0, 0.25, and 1. The detected column above counts all 100 panels, so the two rates are not the same denominator. Paired probabilities are a Normal restatement of each audit panel's own posterior mean and standard deviation. Agreement with the audit rate is not validation and is not transport.

| Scenario | Curve p_detect | Audit rates | Paired mean p_detect | Mean-SD correlation | Divergences | Status |
| --- | --- | --- | --- | --- | --- | --- |
| correctly_specified_noisy | 0.365, 0.604, 0.962 | 0.00=0.360, 0.25=0.540, 1.00=0.980 | 0.360, 0.540, 0.980 | -0.220 | 0 | model-conditional |
| persistent_level_trend_misfit | 1.000, 1.000, 1.000 | 0.00=1.000, 0.25=1.000, 1.00=1.000 | 1.000, 1.000, 1.000 | -0.052 | 0 | model-conditional |
| serially_correlated_errors | 0.408, 0.515, 0.770 | 0.00=0.380, 0.25=0.520, 1.00=0.840 | 0.360, 0.520, 0.840 | -0.076 | 0 | model-conditional |

## Historical replay

Evidence level: historical-descriptive. Selection windows detected 2 of 3 scored windows at the zero lift. Binomial standard error: None. Audit fit: False. Calibration evidence level: model-conditional. Curve validated: False.

| Origin | Role | Status | Detected |
| --- | --- | --- | --- |
| 16 | selection | scored | False |
| 20 | selection | scored | True |
| 24 | selection | scored | True |
| 32 | audit | reserved | None |

## Claims

Lift 0.00. Under protocol issue-914-qualification-v1, scenario 'correctly_specified_noisy', estimator 'synthetic_control', and the shipped positive cumulative decision rule, the study reports the observed detection count with failures included in the denominator of 100 requested panels. 0 fit failures are non-detections, not omitted panels. This is not a launch recommendation, and it is not a claim that the detection target of 0.8 was achieved.

Recovery at lift 0.00. The Monte Carlo bias interval covers zero. Unbiasedness is withheld. No bias tolerance was predeclared. 95 convex-hull warnings are recorded, so this is not a correctly specified recovery diagnostic. This is not a launch recommendation.

Lift 0.25. Under protocol issue-914-qualification-v1, scenario 'correctly_specified_noisy', estimator 'synthetic_control', and the shipped positive cumulative decision rule, the study reports the observed detection count with failures included in the denominator of 100 requested panels. 0 fit failures are non-detections, not omitted panels. This is not a launch recommendation, and it is not a claim that the detection target of 0.8 was achieved.

Recovery at lift 0.25. The Monte Carlo bias interval covers zero. Unbiasedness is withheld. No bias tolerance was predeclared. 95 convex-hull warnings are recorded, so this is not a correctly specified recovery diagnostic. This is not a launch recommendation.

Lift 1.00. Under protocol issue-914-qualification-v1, scenario 'correctly_specified_noisy', estimator 'synthetic_control', and the shipped positive cumulative decision rule, the study reports the observed detection count with failures included in the denominator of 100 requested panels. 0 fit failures are non-detections, not omitted panels. This is not a launch recommendation, and it is not a claim that the detection target of 0.8 was achieved.

Recovery at lift 1.00. The Monte Carlo bias interval covers zero. Unbiasedness is withheld. No bias tolerance was predeclared. 95 convex-hull warnings are recorded, so this is not a correctly specified recovery diagnostic. This is not a launch recommendation.

Model-conditional comparison under the named assumptions: rope_half_width=0 so the engine positive rule matches the shipped tail rule, threshold is the 0.95 tail probability, not the effect threshold of 0, fold SDs are posterior SDs of cumulative impact, not residual SDs, the engine crosses every null draw with every fold SD, a calibration curve does not transport by itself, the learned-null fitter floors fold SDs below 1e-6 internally; the record stores raw SDs. Numerical agreement is not validation, and this is not a launch recommendation.

Lift 0.00. Under protocol issue-914-qualification-v1, scenario 'persistent_level_trend_misfit', estimator 'synthetic_control', and the shipped positive cumulative decision rule, the study reports the observed detection count with failures included in the denominator of 100 requested panels. 0 fit failures are non-detections, not omitted panels. This is not a launch recommendation, and it is not a claim that the detection target of 0.8 was achieved. This scenario is the persistent level and trend misfit. The count describes that biased process. It is not evidence that the estimator recovered an effect.

Recovery at lift 0.00. The Monte Carlo bias interval does not cover zero. Unbiasedness is withheld. No bias tolerance was predeclared. This scenario is the persistent level and trend misfit, so the interval is not a recovery diagnostic. This is not a launch recommendation.

Lift 0.25. Under protocol issue-914-qualification-v1, scenario 'persistent_level_trend_misfit', estimator 'synthetic_control', and the shipped positive cumulative decision rule, the study reports the observed detection count with failures included in the denominator of 100 requested panels. 0 fit failures are non-detections, not omitted panels. This is not a launch recommendation, and it is not a claim that the detection target of 0.8 was achieved. This scenario is the persistent level and trend misfit. The count describes that biased process. It is not evidence that the estimator recovered an effect.

Recovery at lift 0.25. The Monte Carlo bias interval does not cover zero. Unbiasedness is withheld. No bias tolerance was predeclared. This scenario is the persistent level and trend misfit, so the interval is not a recovery diagnostic. This is not a launch recommendation.

Lift 1.00. Under protocol issue-914-qualification-v1, scenario 'persistent_level_trend_misfit', estimator 'synthetic_control', and the shipped positive cumulative decision rule, the study reports the observed detection count with failures included in the denominator of 100 requested panels. 0 fit failures are non-detections, not omitted panels. This is not a launch recommendation, and it is not a claim that the detection target of 0.8 was achieved. This scenario is the persistent level and trend misfit. The count describes that biased process. It is not evidence that the estimator recovered an effect.

Recovery at lift 1.00. The Monte Carlo bias interval does not cover zero. Unbiasedness is withheld. No bias tolerance was predeclared. This scenario is the persistent level and trend misfit, so the interval is not a recovery diagnostic. This is not a launch recommendation.

Model-conditional comparison under the named assumptions: rope_half_width=0 so the engine positive rule matches the shipped tail rule, threshold is the 0.95 tail probability, not the effect threshold of 0, fold SDs are posterior SDs of cumulative impact, not residual SDs, the engine crosses every null draw with every fold SD, a calibration curve does not transport by itself, the learned-null fitter floors fold SDs below 1e-6 internally; the record stores raw SDs. Numerical agreement is not validation, and this is not a launch recommendation.

Lift 0.00. Under protocol issue-914-qualification-v1, scenario 'serially_correlated_errors', estimator 'synthetic_control', and the shipped positive cumulative decision rule, the study reports the observed detection count with failures included in the denominator of 100 requested panels. 0 fit failures are non-detections, not omitted panels. This is not a launch recommendation, and it is not a claim that the detection target of 0.8 was achieved.

Recovery at lift 0.00. The Monte Carlo bias interval covers zero. Unbiasedness is withheld. No bias tolerance was predeclared. This scenario is not the reference scenario, so the interval is not a recovery diagnostic. This is not a launch recommendation.

Lift 0.25. Under protocol issue-914-qualification-v1, scenario 'serially_correlated_errors', estimator 'synthetic_control', and the shipped positive cumulative decision rule, the study reports the observed detection count with failures included in the denominator of 100 requested panels. 0 fit failures are non-detections, not omitted panels. This is not a launch recommendation, and it is not a claim that the detection target of 0.8 was achieved.

Recovery at lift 0.25. The Monte Carlo bias interval covers zero. Unbiasedness is withheld. No bias tolerance was predeclared. This scenario is not the reference scenario, so the interval is not a recovery diagnostic. This is not a launch recommendation.

Lift 1.00. Under protocol issue-914-qualification-v1, scenario 'serially_correlated_errors', estimator 'synthetic_control', and the shipped positive cumulative decision rule, the study reports the observed detection count with failures included in the denominator of 100 requested panels. 0 fit failures are non-detections, not omitted panels. This is not a launch recommendation, and it is not a claim that the detection target of 0.8 was achieved.

Recovery at lift 1.00. The Monte Carlo bias interval covers zero. Unbiasedness is withheld. No bias tolerance was predeclared. This scenario is not the reference scenario, so the interval is not a recovery diagnostic. This is not a launch recommendation.

Model-conditional comparison under the named assumptions: rope_half_width=0 so the engine positive rule matches the shipped tail rule, threshold is the 0.95 tail probability, not the effect threshold of 0, fold SDs are posterior SDs of cumulative impact, not residual SDs, the engine crosses every null draw with every fold SD, a calibration curve does not transport by itself, the learned-null fitter floors fold SDs below 1e-6 internally; the record stores raw SDs. Numerical agreement is not validation, and this is not a launch recommendation.

Historical replay is descriptive product evidence. It does not carry a binomial standard error, and it is not a launch recommendation.

Numeric agreement between a curve and a rate is not validation.

Source SHA: c4fa86726b040ba7164b9396eb5849c90d271b37.
Product modules were at that commit. Qualification modules were untracked when the draws were generated:

```text
?? causalpy/tests/qualification_panels.py
?? causalpy/tests/qualification_policy.py
?? causalpy/tests/qualification_protocol.py
?? causalpy/tests/qualification_study.py
?? causalpy/tests/test_qualification.py
```

Eligibility labels were re-derived from these stored counts after a policy correction. Recovery claims withhold unbiasedness: no bias tolerance was predeclared, and the reference hull-warning count confounds that label. The panel draws were not refit.
