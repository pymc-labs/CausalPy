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
![Interrogate](docs/source/_static/interrogate_badge.svg)
[![codecov](https://codecov.io/gh/pymc-labs/CausalPy/branch/main/graph/badge.svg?token=FDKNAY5CZ9)](https://codecov.io/gh/pymc-labs/CausalPy)

# CausalPy

A Python package focussing on causal inference in quasi-experimental settings. The package allows for sophisticated Bayesian model fitting methods to be used in addition to traditional OLS.

## Installation

To get the latest release:
```bash
pip install CausalPy
```

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

# Visualize outputs
fig, ax = result.plot();

# Get a results summary
result.summary()

plt.show()
```

## Roadmap

Plans for the repository can be seen in the [Issues](https://github.com/pymc-labs/CausalPy/issues).

## Videos
Click on the thumbnail below to watch a video about CausalPy on YouTube.
[![Youtube video thumbnail image](https://img.youtube.com/vi/gV6wzTk3o1U/maxresdefault.jpg)](https://www.youtube.com/watch?v=gV6wzTk3o1U)

## Features

CausalPy has a broad range of quasi-experimental methods for causal inference:

| Method	| Description |
|-|-|
| Synthetic control | Constructs a synthetic version of the treatment group from a weighted combination of control units. Used for causal inference in comparative case studies when a single unit is treated, and there are multiple control units.|
| Geographical lift | Measures the impact of an intervention in a specific geographic area by comparing it to similar areas without the intervention. Commonly used in marketing to assess regional campaigns. |
| ANCOVA | Analysis of Covariance combines ANOVA and regression to control for the effects of one or more quantitative covariates. Used when comparing group means while controlling for other variables. |
| Differences in Differences | Compares the changes in outcomes over time between a treatment group and a control group. Used in observational studies to estimate causal effects by accounting for time trends. |
| Regression discontinuity | Identifies causal effects by exploiting a cutoff or threshold in an assignment variable. Used when treatment is assigned based on a threshold value of an observed variable, allowing comparison just above and below the cutoff. |
| Regression kink designs | Focuses on changes in the slope (kinks) of the relationship between variables rather than jumps at cutoff points. Used to identify causal effects when treatment intensity changes at a threshold. |
| Interrupted time series | Analyzes the effect of an intervention by comparing time series data before and after the intervention. Used when data is collected over time and an intervention occurs at a known point, allowing assessment of changes in level or trend. |
| Instrumental variable regression | Addresses endogeneity by using an instrument variable that is correlated with the endogenous explanatory variable but uncorrelated with the error term. Used when explanatory variables are correlated with the error term, providing consistent estimates of causal effects. |
| Inverse Propensity Score Weighting | Weights observations by the inverse of the probability of receiving the treatment. Used in causal inference to create a synthetic sample where the treatment assignment is independent of measured covariates, helping to adjust for confounding variables in observational studies. |

## License

[Apache License 2.0](LICENSE)

---

## Support

<img src="https://raw.githubusercontent.com/pymc-labs/CausalPy/main/docs/source/_static/pymc-labs-log.png" align="right" width="50%" />

This repository is supported by [PyMC Labs](https://www.pymc-labs.com).

If you are interested in seeing what PyMC Labs can do for you, then please email [ben.vincent@pymc-labs.com](mailto:ben.vincent@pymc-labs.com). We work with companies at a variety of scales and with varying levels of existing modeling capacity. We also run corporate workshop training events and can provide sessions ranging from introduction to Bayes to more advanced topics.
