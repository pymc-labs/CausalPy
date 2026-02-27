:::{image} _static/logo.png
:width: 60 %
:align: center
:alt: CausalPy logo
:::

# CausalPy - causal inference for quasi-experiments

A Python package focussing on causal inference for quasi-experiments. The package allows users to use different model types. Sophisticated Bayesian methods can be used, harnessing the power of [PyMC](https://www.pymc.io/) and [ArviZ](https://python.arviz.org). But users can also use more traditional [Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) estimation methods via [scikit-learn](https://scikit-learn.org) models.

## Installation

To get the latest release you can use pip:

```bash
pip install CausalPy
```

or conda/mamba/micromamba:

```bash
conda install causalpy -c conda-forge    # or mamba/micromamba
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
fig, ax = result.plot()
# Get a results summary
result.summary()

plt.show()
```

## Videos

<style>
.video-container {
    position: relative;
    padding-bottom: 56.25%; /* 16:9 aspect ratio */
    height: 0;
    overflow: hidden;
    max-width: 100%;
    background: #000;
}

.video-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: 0;
}
</style>

<div class="video-container">
    <iframe src="https://www.youtube.com/embed/gV6wzTk3o1U" title="YouTube video player" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

## Features
CausalPy has a broad range of quasi-experimental methods for causal inference:

| Method	| Description |
|-|-|
| Synthetic control | Constructs a synthetic version of the treatment group from a weighted combination of control units. Used for causal inference in comparative case studies when a single unit is treated, and there are multiple control units.|
| Geographical lift | Measures the impact of an intervention in a specific geographic area by comparing it to similar areas without the intervention. Commonly used in marketing to assess regional campaigns. |
| ANCOVA | Analysis of Covariance combines ANOVA and regression to control for the effects of one or more quantitative covariates. Used when comparing group means while controlling for other variables. |
| Differences in Differences | Compares the changes in outcomes over time between a treatment group and a control group. Used in observational studies to estimate causal effects by accounting for time trends. |
|Regression discontinuity | Identifies causal effects by exploiting a sharp cutoff or threshold in an assignment variable. Used when treatment is assigned based on a threshold value of an observed variable, allowing comparison just above and below the cutoff. |
| Regression kink designs | Focuses on changes in the slope (kinks) of the relationship between variables rather than jumps at cutoff points. Used to identify causal effects when treatment intensity changes at a threshold. |
| Interrupted time series | Analyzes the effect of an intervention by comparing time series data before and after the intervention. Used when data is collected over time and an intervention occurs at a known point, allowing assessment of changes in level or trend. |
| Instrumental variable regression | Addresses endogeneity by using an instrument variable that is correlated with the endogenous explanatory variable but uncorrelated with the error term. Used when explanatory variables are correlated with the error term, providing consistent estimates of causal effects. |
| Inverse Propensity Score Weighting | Weights observations by the inverse of the probability of receiving the treatment. Used in causal inference to create a synthetic sample where the treatment assignment is independent of measured covariates, helping to adjust for confounding variables in observational studies. |

## Getting Help

Have questions about using CausalPy? We're here to help!

- **Questions & Help**: Visit our [GitHub Discussions Q&A](https://github.com/pymc-labs/CausalPy/discussions/categories/q-a) to ask questions and get help from the community
- **Bug Reports & Feature Requests**: Open an [Issue](https://github.com/pymc-labs/CausalPy/issues) for bugs or feature requests
- **Documentation**: Browse the [knowledgebase](knowledgebase/index), [API documentation](api/index), and [examples](notebooks/index) for detailed guides

Please use GitHub Discussions for general questions rather than opening issues, so we can keep the issue tracker focused on bugs and enhancements.

## Support

This repository is supported by [PyMC Labs](https://www.pymc-labs.io).

For companies that want to use CausalPy in production, [PyMC Labs](https://www.pymc-labs.com) is available for consulting and training. We can help you build and deploy your models in production. We have experience with cutting edge Bayesian and causal modelling techniques which we have applied to a range of business domains.

<p align="center">
  <a href="https://www.pymc-labs.io">
    <img src="./_static/pymc-labs-log.png" alt="PyMC Labs Logo" style="width:50%;">
  </a>
</p>

:::{toctree}
:hidden:

knowledgebase/index
api/index
notebooks/index
:::
