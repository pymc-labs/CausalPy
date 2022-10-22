# QuasPy

A Python package focussing on causal inference in a number of quasi-experimental settings. The package allows for both Bayesian and traditional model fitting methods.

We cover the following quasi-experimental situations and models:

| method                           | `scikit-learn` models  | `pymc` models |
|----------------------------------|-----------------------|---------------|
| Synthetic control                | ✅                     | ⏳             |
| Interrupted time series          | ✅                     | ⏳             |
| Difference in differences        | ✅                     | ⏳             |
| Regression discontinuity designs | ✅                     | ❌             |
|                                  | [`scikit-learn` examples](notebooks/skl_demos.ipynb) |               |

## Comparison to related packages
|                           | [CausalImpact](https://google.github.io/CausalImpact/) from Google | [GeoLift](https://github.com/facebookincubator/GeoLift/) from Meta | QuasPy from PyMC Labs                                |
|---------------------------|--------------------------------|---------|----------------------------------------|
| interrupted time series   | ✅                              | ❌       | ✅                                      |
| synthetic control         | ❌                              | ✅       | ✅                                      |
| regression discontinuity  | ❌                              | ❌       | ✅                                      |
| difference in differences | ❌                              | ❌       | ✅                                      |
| Language                  | R (but see [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact))  | R       | Python                                 |
| Models                    | Bayesian structural timeseries | Augmented synthetic control      | Flexible Bayesian & traditional models |

## Synthetic control
This is appropriate when you have multiple units, one of which is treated. You build a synthetic control as a weighted combination of the untreated units.

| time | Treatment | Control 1 | Control 2 | Control 3 |
|------|-----------|-----------|-----------|-----------|
| 0    | $y_0$ | $x_{1,0}$ | $x_{2,0}$ | $x_{3,0}$ |
| 1    | $y_1$ | $x_{1,1}$ | $x_{2,1}$ | $x_{3,1}$ |
|$\ldots$ | $\ldots$  | $\ldots$  | $\ldots$  | $\ldots$  |
| N    | $y_N$ | $x_{1,N}$ | $x_{2,N}$ | $x_{3,N}$ |

A worked example is given in the [Synthetic control](notebooks/synthetic_control.ipynb) notebook.

![](img/synthetic_control_skl.png)

## Interrupted time series
This is appropriate when you have a single treated unit, and therefore a single time series, and do _not_ have a set of untreated units.

| time | Treatment |
|------|-----------|
| 0    | $y_0$ |
| 1    | $y_1$ |
|$\ldots$ | $\ldots$  |
| N    | $y_N$ |

A worked example is given in the [Interrupted time series](notebooks/interrupted_time_series_no_predictors.ipynb) notebook.

![](img/interrupted_time_series_skl.png)

## Difference in Differences

This is appropriate when you have a single pre and post intervention measurement and have a treament and a control group.

Data is expected to be in the following form. Shown are just two units - one in the treated group (`group=1`) and one in the untreated group (`group=0`), but there can of course be multiple units per group. This is panel data (also known as repeated measures) where each unit is measured at 2 time points.

| unit | t | group | y         |
|------|---|-------|-----------|
| 0    | 0 | 0     | $y_{0,0}$ |
| 0    | 1 | 0     | $y_{0,0}$ |
| 1    | 0 | 1     | $y_{1,0}$ |
| 1    | 1 | 1     | $y_{1,1}$ |

![](img/difference_in_differences_skl.png)

## Regression discontinuity designs

Regression discontinuity designs are used when treatment is applied to units according to a cutoff on the running variable (e.g. $x$). By looking for the presence of a discontinuity at the precise point of the treatment cutoff then we can make causal claims about the potential impact of the treatment.

| x         | y         | treated  |
|-----------|-----------|----------|
| $x_0$     | $y_0$     | False    |
| $x_1$     | $y_0$     | False    |
| $\ldots$  | $\ldots$  | $\ldots$ |
| $x_{N-1}$ | $y_{N-1}$ | True     |
| $x_N$     | $y_N$     | True     |

![](img/regression_discontinuity_skl.png)

## Related packages

* [CausalImpact](https://google.github.io/CausalImpact/) from Google
* [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact)
* [GeoLift](https://github.com/facebookincubator/GeoLift/) by Meta


## Learning resources

Here are some general resources about causal inference:

* The official [PyMC examples gallery](https://www.pymc.io/projects/examples/en/latest/gallery.html) has a set of examples specifically relating to causal inference.
* Huntington-Klein, N. (2021). [The effect: An introduction to research design and causality](https://theeffectbook.net). Chapman and Hall/CRC.
* Cunningham, S. (2021). [Causal inference: The Mixtape](https://mixtape.scunning.com). Yale University Press.

## Installation

[coming soon]

--- 

## Local development

1. Create conda environment:

```bash
conda create --name causal_impact_env --file requirements.txt
```

2. Activate environment:

```bash
conda activate causal_impact_env
```

3. Import the package

```bash
pip install -e ./
```