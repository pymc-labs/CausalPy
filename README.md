<div align="center">
  <a href="https://github.com/pymc-labs/CausalPy"><img width="60%" src="https://raw.githubusercontent.com/pymc-labs/CausalPy/main/docs/source/_static/logo.png"></a>
</div>

----

![Build Status](https://github.com/pymc-labs/CausalPy/actions/workflows/ci.yml/badge.svg?branch=main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://badge.fury.io/py/CausalPy.svg)](https://badge.fury.io/py/CausalPy)
![GitHub Repo stars](https://img.shields.io/github/stars/pymc-labs/causalpy?style=social)
![Read the Docs](https://img.shields.io/readthedocs/causalpy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/causalpy)
![Interrogate](https://raw.githubusercontent.com/pymc-labs/CausalPy/interrogate-badges/interrogate_badge.svg)
[![codecov](https://codecov.io/gh/pymc-labs/CausalPy/branch/main/graph/badge.svg?token=FDKNAY5CZ9)](https://codecov.io/gh/pymc-labs/CausalPy)

# CausalPy

**Research-grade causal inference workflows** for quasi-experimental designs in Python.

CausalPy helps you estimate causal effects with transparent assumptions, uncertainty-aware modeling, and reproducible outputs:

- **Quasi-experimental methods:** Difference-in-differences, synthetic control, regression discontinuity, interrupted time series, instrumental variables, and more
- **Bayesian-first estimation** via [PyMC](https://www.pymc.io/) with full uncertainty quantification, plus traditional OLS via [scikit-learn](https://scikit-learn.org)
- **Decision-ready outputs:** Effect summaries with credible intervals (HDI), practical significance (ROPE), and publication-quality plots

**Non-goals:** CausalPy is a research and analysis library, not a workflow orchestration or governance platform (no pipelines, scheduling, permissions, or registries).

## Installation

To get the latest release:

```bash
pip install CausalPy
```

If you run into installation issues with PyMC (e.g. BLAS or compilation), try the conda install: `conda install -c conda-forge causalpy`.

Alternatively, if you want the very latest version of the package you can install from GitHub:

```bash
pip install git+https://github.com/pymc-labs/CausalPy.git
```

## Quickstart

```python
import causalpy as cp
import matplotlib.pyplot as plt

# Import and process data
df = (
    cp.load_data("drinking")
    .rename(columns={"agecell": "age"})
    .assign(treated=lambda df_: df_.age > 21)
)

# Run the analysis
result = cp.RegressionDiscontinuity(
    df,
    formula="all ~ 1 + age + treated",
    running_variable_name="age",
    model=cp.pymc_models.LinearRegression(),
    treatment_threshold=21,
)

# Visualize the causal effect at the threshold
fig, ax = result.plot()

# Get a results summary with posterior estimates
result.summary()
```

The `result.plot()` visualizes the regression discontinuity design, showing the estimated jump at the treatment threshold. The `result.summary()` prints posterior estimates of the causal effect with uncertainty intervals.

## Videos

Click on the thumbnail below to watch a video about CausalPy on YouTube.

[![Youtube video thumbnail image](https://img.youtube.com/vi/gV6wzTk3o1U/maxresdefault.jpg)](https://www.youtube.com/watch?v=gV6wzTk3o1U)

## When CausalPy is a good fit

- You have a plausible quasi-experimental design (threshold rule, policy change, staggered rollout, geo lift, etc.)
- You want uncertainty-aware estimates and diagnostics, not only point estimates
- You need reproducible analysis artifacts for review and communication

## When CausalPy is not a fit

- You need causal discovery from weakly identified observational data
- You want fully automated "black box" causal answers without specifying assumptions
- You primarily need production workflow tooling (pipelines, governance, multi-user collaboration)

## Methods and Workflows

CausalPy provides methods for common causal inference decision contexts:

| Decision context | Methods |
|------------------|---------|
| Comparative case studies | Synthetic control, Geographical lift |
| Policy/rollout evaluation | Differences in Differences, Staggered DiD, Interrupted time series |
| Threshold assignment | Regression discontinuity, Regression kink |
| Confounding/endogeneity | Instrumental variables, Inverse propensity weighting |
| Covariate adjustment | ANCOVA |

### Available methods

| Method | Description |
|-|-|
| Synthetic control | Constructs a synthetic version of the treatment group from a weighted combination of control units. Used for causal inference in comparative case studies when a single unit is treated, and there are multiple control units. |
| Geographical lift | Measures the impact of an intervention in a specific geographic area by comparing it to similar areas without the intervention. Commonly used in marketing to assess regional campaigns. |
| ANCOVA | Analysis of Covariance combines ANOVA and regression to control for the effects of one or more quantitative covariates. Used when comparing group means while controlling for other variables. |
| Differences in Differences | Compares the changes in outcomes over time between a treatment group and a control group. Used in observational studies to estimate causal effects by accounting for time trends. |
| Staggered Difference-in-Differences | Estimates event-time treatment effects when different units adopt treatment at different times, using an imputation approach that models untreated outcomes and compares observed outcomes to counterfactual predictions. |
| Regression discontinuity | Identifies causal effects by exploiting a cutoff or threshold in an assignment variable. Used when treatment is assigned based on a threshold value of an observed variable, allowing comparison just above and below the cutoff. |
| Regression kink designs | Focuses on changes in the slope (kinks) of the relationship between variables rather than jumps at cutoff points. Used to identify causal effects when treatment intensity changes at a threshold. |
| Interrupted time series | Analyzes the effect of an intervention by comparing time series data before and after the intervention. Used when data is collected over time and an intervention occurs at a known point, allowing assessment of changes in level or trend. |
| Instrumental variable regression | Addresses endogeneity by using an instrument variable that is correlated with the endogenous explanatory variable but uncorrelated with the error term. Used when explanatory variables are correlated with the error term, providing consistent estimates of causal effects. |
| Inverse Propensity Score Weighting | Weights observations by the inverse of the probability of receiving the treatment. Used in causal inference to create a synthetic sample where the treatment assignment is independent of measured covariates, helping to adjust for confounding variables in observational studies. |

## Diagnostics-first by design

CausalPy emphasizes transparent, uncertainty-aware outputs for rigorous causal analysis:

- **Effect summaries:** Every experiment provides `effect_summary()` returning decision-ready statistics with both tabular and prose formats
- **Uncertainty quantification:** Bayesian models report HDI (Highest Density Intervals); OLS models report confidence intervals
- **Practical significance:** ROPE (Region of Practical Equivalence) analysis to assess whether effects exceed meaningful thresholds
- **Direction testing:** Tail probabilities (e.g., P(effect > 0)) for directional inference

## Consulting

<img src="https://raw.githubusercontent.com/pymc-labs/CausalPy/main/docs/source/_static/pymc-labs-log.png" align="right" width="40%" />

**Need expert help with causal inference?** [PyMC Labs](https://www.pymc-labs.com) offers:

- **Causal Design Reviews** (1-2 weeks): Identification strategy assessment, diagnostics plan, and reproducible analysis report
- **Implementation engagements**: End-to-end support for high-stakes causal analyses
- **Training**: Corporate workshops from Bayesian fundamentals to advanced causal inference

Contact [ben.vincent@pymc-labs.com](mailto:ben.vincent@pymc-labs.com) to discuss your needs.

## Citing CausalPy

If you use CausalPy in your research, please cite it. A Zenodo DOI for stable releases is planned. In the meantime, you can cite the repository:

```
@software{causalpy,
  author = {{PyMC Labs}},
  title = {CausalPy: Causal inference for quasi-experiments in Python},
  url = {https://github.com/pymc-labs/CausalPy},
  year = {2024}
}
```

## Roadmap

Plans for the repository can be seen in the [Issues](https://github.com/pymc-labs/CausalPy/issues).

## License

[Apache License 2.0](LICENSE)

---

## Getting Help

Have questions about using CausalPy? We're here to help!

- **Questions & Help**: Visit our [GitHub Discussions Q&A](https://github.com/pymc-labs/CausalPy/discussions/categories/q-a) to ask questions and get help from the community
- **Bug Reports & Feature Requests**: Open an [Issue](https://github.com/pymc-labs/CausalPy/issues) for bugs or feature requests
- **Documentation**: Check out our [documentation](https://causalpy.readthedocs.io) for detailed guides and examples

Please use GitHub Discussions for general questions rather than opening issues, so we can keep the issue tracker focused on bugs and enhancements.

## Support

This repository is supported by [PyMC Labs](https://www.pymc-labs.com).

If you are interested in seeing what PyMC Labs can do for you, then please email [ben.vincent@pymc-labs.com](mailto:ben.vincent@pymc-labs.com). We work with companies at a variety of scales and with varying levels of existing modeling capacity. We also run corporate workshop training events and can provide sessions ranging from introduction to Bayes to more advanced topics.
