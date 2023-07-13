<div align="center">
  <a href="https://github.com/pymc-labs/CausalPy"><img width="60%" src="docs/source/_static/logo.png"></a>
</div>

----

![Build](https://github.com/pymc-labs/CausalPy/workflows/ci/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/CausalPy.svg)](https://badge.fury.io/py/CausalPy)
![GitHub Repo stars](https://img.shields.io/github/stars/pymc-labs/causalpy?style=social)
![Read the Docs](https://img.shields.io/readthedocs/causalpy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/causalpy)
![Interrogate](docs/source/_static/interrogate_badge.svg)
[![codecov](https://codecov.io/gh/pymc-labs/CausalPy/branch/main/graph/badge.svg?token=FDKNAY5CZ9)](https://codecov.io/gh/pymc-labs/CausalPy)

# CausalPy

A Python package focussing on causal inference in quasi-experimental settings. The package allows for sophisticated Bayesian model fitting methods to be used in addition to traditional OLS.

## Comparison to related packages

Rather than focussing on one particular quasi-experimental setting, this package aims to have broad applicability.

Another distinctive feature of this package is the ability to use different models. Currently, users can fit with `scikit-learn` models or Bayesian models with `PyMC`.

|                           | [CausalImpact](https://google.github.io/CausalImpact/) from Google | [GeoLift](https://github.com/facebookincubator/GeoLift/) from Meta | CausalPy from [PyMC Labs](https://www.pymc-labs.io) |
|---------------------------|--------------------------------|---------|----------------------------------------|
| Synthetic control         | ✅                              | ✅       | ✅                                      |
| Regression discontinuity  | ❌                              | ❌       | ✅                                      |
| Difference in differences | ❌                              | ❌       | ✅                                      |
| Language                  | R (but see [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact))  | R       | Python                                 |
| Models                    | Bayesian structural timeseries | Augmented synthetic control      | Flexible: Traditional OLS and Bayesian models |

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


# Import and process data
df = (
    cp.load_data("drinking")
    .rename(columns={"agecell": "age"})
    .assign(treated=lambda df_: df_.age > 21)
    )

# Run the analysis
result = cp.pymc_experiments.RegressionDiscontinuity(
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
```

## Roadmap

Plans for the repository can be seen in the [Issues](https://github.com/pymc-labs/CausalPy/issues).

## Videos
Click on the thumbnail below to watch a video about CausalPy on YouTube.
[![Youtube video thumbnail image](https://img.youtube.com/vi/gV6wzTk3o1U/maxresdefault.jpg)](https://www.youtube.com/watch?v=gV6wzTk3o1U)

## Overview of package capabilities

### Synthetic control
This is appropriate when you have multiple units, one of which is treated. You build a synthetic control as a weighted combination of the untreated units.

| Time | Outcome   | Control 1 | Control 2 | Control 3 |
|------|-----------|-----------|-----------|-----------|
| 0    | $y_0$ | $x_{1,0}$ | $x_{2,0}$ | $x_{3,0}$ |
| 1    | $y_1$ | $x_{1,1}$ | $x_{2,1}$ | $x_{3,1}$ |
|$\ldots$ | $\ldots$  | $\ldots$  | $\ldots$  | $\ldots$  |
| T    | $y_T$ | $x_{1,T}$ | $x_{2,T}$ | $x_{3,T}$ |


| Frequentist | Bayesian |
|--|--|
| ![](docs/source/_static/synthetic_control_skl.svg) | ![](docs/source/_static/synthetic_control_pymc.svg) |

> The data (treated and untreated units), pre-treatment model fit, and counterfactual (i.e. the synthetic control) are plotted (top). The causal impact is shown as a blue shaded region. The Bayesian analysis shows shaded Bayesian credible regions of the model fit and counterfactual. Also shown is the causal impact (middle) and cumulative causal impact (bottom).

### Geographical lift (Geolift)
We can also use synthetic control methods to analyse data from geographical lift studies. For example, we can try to evaluate the causal impact of an intervention (e.g. a marketing campaign) run in one geographical area by using control geographical areas which are similar to the intervention area but which did not recieve the specific marketing intervention.

### ANCOVA

This is appropriate for non-equivalent group designs when you have a single pre and post intervention measurement and have a treament and a control group.

| Group | pre | post |
|------|---|-------|
| 0    | $x_1$ | $y_1$ |
| 0    | $x_2$ | $y_2$ |
| 1    | $x_3$ | $y_3$ |
| 1    | $x_4$ | $y_4$ |

| Frequentist | Bayesian |
|--|--|
| coming soon | ![](docs/source/_static/anova_pymc.svg) |

> The data from the control and treatment group are plotted, along with posterior predictive 94% credible intervals. The lower panel shows the estimated treatment effect.

### Difference in Differences

This is appropriate for non-equivalent group designs when you have pre and post intervention measurement and have a treament and a control group. Unlike the ANCOVA approach, difference in differences is appropriate when there are multiple pre and/or post treatment measurements.

Data is expected to be in the following form. Shown are just two units - one in the treated group (`group=1`) and one in the untreated group (`group=0`), but there can of course be multiple units per group. This is panel data (also known as repeated measures) where each unit is measured at 2 time points.

| Unit | Time | Group | Outcome         |
|------|---|-------|-----------|
| 0    | 0 | 0     | $y_{0,0}$ |
| 0    | 1 | 0     | $y_{0,0}$ |
| 1    | 0 | 1     | $y_{1,0}$ |
| 1    | 1 | 1     | $y_{1,1}$ |

| Frequentist | Bayesian |
|--|--|
| ![](docs/source/_static/difference_in_differences_skl.svg) | ![](docs/source/_static/difference_in_differences_pymc.svg) |

>The data, model fit, and counterfactual are plotted. Frequentist model fits result in points estimates, but the Bayesian analysis results in posterior distributions, represented by the violin plots. The causal impact is the difference between the counterfactual prediction (treated group, post treatment) and the observed values for the treated group, post treatment.

### Regression discontinuity designs

Regression discontinuity designs are used when treatment is applied to units according to a cutoff on the running variable (e.g. $x$) which is typically _not_ time. By looking for the presence of a discontinuity at the precise point of the treatment cutoff then we can make causal claims about the potential impact of the treatment.

| Running variable | Outcome | Treated  |
|-----------|-----------|----------|
| $x_0$     | $y_0$     | False    |
| $x_1$     | $y_0$     | False    |
| $\ldots$  | $\ldots$  | $\ldots$ |
| $x_{N-1}$ | $y_{N-1}$ | True     |
| $x_N$     | $y_N$     | True     |


| Frequentist | Bayesian |
|--|--|
| ![](docs/source/_static/regression_discontinuity_skl.svg) | ![](docs/source/_static/regression_discontinuity_pymc.svg) |

> The data, model fit, and counterfactual are plotted (top). Frequentist analysis shows the causal impact with the blue shaded region, but this is not shown in the Bayesian analysis to avoid a cluttered chart. Instead, the Bayesian analysis shows shaded Bayesian credible regions of the model fits. The Frequentist analysis visualises the point estimate of the causal impact, but the Bayesian analysis also plots the posterior distribution of the regression discontinuity effect (bottom).

### Interrupted time series

Interrupted time series analysis is appropriate when you have a time series of observations which undergo treatment at a particular point in time. This kind of analysis has no control group and looks for the presence of a change in the outcome measure at or soon after the treatment time. Multiple predictors can be included.

| Time | Outcome | Treated  | Predictor |
|-----------|-----------|----------|----------|
| $t_0$     | $y_0$     | False    | $x_0$     |
| $t_1$     | $y_0$     | False    | $x_1$     |
| $\ldots$  | $\ldots$  | $\ldots$ | $\ldots$  |
| $t_{N-1}$ | $y_{N-1}$ | True     | $x_{N-1}$ |
| $t_N$     | $y_N$     | True     | $x_N$     |


| Frequentist | Bayesian |
|--|--|
| coming soon | ![](docs/source/_static/interrupted_time_series_pymc.svg) |

> The data, pre-treatment model fit, and counterfactual are plotted (top). The causal impact is shown as a blue shaded region. The Bayesian analysis shows shaded Bayesian credible regions of the model fit and counterfactual. Also shown is the causal impact (middle) and cumulative causal impact (bottom).

## Learning resources

Here are some general resources about causal inference:

* The official [PyMC examples gallery](https://www.pymc.io/projects/examples/en/latest/gallery.html) has a set of examples specifically relating to causal inference.
* Angrist, J. D., & Pischke, J. S. (2009). Mostly harmless econometrics: An empiricist's companion. Princeton university press.
* Angrist, J. D., & Pischke, J. S. (2014). Mastering'metrics: The path from cause to effect. Princeton university press.
* Cunningham, S. (2021). [Causal inference: The Mixtape](https://mixtape.scunning.com). Yale University Press.
* Huntington-Klein, N. (2021). [The effect: An introduction to research design and causality](https://theeffectbook.net). Chapman and Hall/CRC.
* Reichardt, C. S. (2019). Quasi-experimentation: A guide to design and analysis. Guilford Publications.

## License

[Apache License 2.0](LICENSE)

---

## Support

<img src="docs/source/_static/pymc-labs-log.png" align="right" width="50%" />

This repository is supported by [PyMC Labs](https://www.pymc-labs.io).

If you are interested in seeing what PyMC Labs can do for you, then please email [ben.vincent@pymc-labs.io](mailto:ben.vincent@pymc-labs.io). We work with companies at a variety of scales and with varying levels of existing modeling capacity. We also run corporate workshop training events and can provide sessions ranging from introduction to Bayes to more advanced topics.
