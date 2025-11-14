# How-to

On this page you can find a gallery of example notebooks that demonstrate the use of CausalPy.

## ANCOVA

Analysis of covariance is a simple linear model, typically with one continuous predictor (the covariate) and a categorical variable (which may correspond to treatment or control group). In the context of this package, ANCOVA could be useful in pre-post treatment designs, either with or without random assignment. This is similar to the approach of difference in differences, but only applicable with a single pre and post treatment measure.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} ANCOVA for pre/post treatment nonequivalent group designs
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/ancova_pymc.png
:link: ancova_pymc
:link-type: doc
:::
::::

## Difference in Differences

Analysis where the treatment effect is estimated as a difference between treatment conditions in the differences between pre-treatment to post treatment observations.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Difference in Differences with `pymc` models
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/did_pymc.png
:link: did_pymc
:link-type: doc
:::
:::{grid-item-card} Banking dataset with a `pymc` model
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/did_pymc_banks.png
:link: did_pymc_banks
:link-type: doc
:::
:::{grid-item-card} Difference in Differences with scikit-learn models
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/did_skl.png
:link: did_skl
:link-type: doc
:::
::::

## Geographical lift testing

Geolift (geographical lift testing) is a method for measuring the causal impact of interventions in geographic regions. It combines synthetic control methods with difference-in-differences approaches to estimate treatment effects when interventions are applied to specific geographic areas.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Bayesian geolift with CausalPy
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/geolift1.png
:link: geolift1
:link-type: doc
:::
:::{grid-item-card} Multi-cell geolift analysis
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/multi_cell_geolift.png
:link: multi_cell_geolift
:link-type: doc
:::
::::

## Instrumental Variables Regression

A quasi-experimental design to estimate a treatment effect where there is a risk of confounding between the treatment and the outcome due to endogeneity. Instrumental variables help identify causal effects by using variables that affect treatment assignment but not the outcome directly.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Instrumental Variable Modelling (IV) with `pymc` models
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/iv_pymc.png
:link: iv_pymc
:link-type: doc
:::
:::{grid-item-card} Instrumental Regression and Justifying Instruments with `pymc`
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/iv_weak_instruments.png
:link: iv_weak_instruments
:link-type: doc
:::
::::

## Interrupted Time Series

A quasi-experimental design that uses time series methods to generate counterfactuals and estimate treatment effects. A series of observations are collected before and after a treatment, and the pre-treatment trend (or any time-series model) is used to predict what would have happened in the absence of treatment.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Excess deaths due to COVID-19
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/its_covid.png
:link: its_covid
:link-type: doc
:::
:::{grid-item-card} Bayesian Interrupted Time Series
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/its_pymc.png
:link: its_pymc
:link-type: doc
:::
:::{grid-item-card} Interrupted Time Series (ITS) with scikit-learn models
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/its_skl.png
:link: its_skl
:link-type: doc
:::
::::

## Inverse Propensity Score Weighting

A method for estimating causal effects by weighting observations by the inverse of their probability of receiving treatment (propensity score). This helps adjust for confounding by creating a pseudo-population where treatment assignment is independent of observed covariates.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} The Paradox of Propensity Scores in Bayesian Inference
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/inv_prop_latent.png
:link: inv_prop_latent
:link-type: doc
:::
:::{grid-item-card} Inverse Propensity Score Weighting with `pymc`
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/inv_prop_pymc.png
:link: inv_prop_pymc
:link-type: doc
:::
::::

## Regression Discontinuity

A quasi-experimental design where treatment assignment is determined by a cutoff point along a running variable (e.g., test score, age, income). The treatment effect is estimated by comparing outcomes just above and below the cutoff, assuming units near the cutoff are similar except for treatment status.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Sharp regression discontinuity with `pymc` models
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/rd_pymc.png
:link: rd_pymc
:link-type: doc
:::
:::{grid-item-card} Drinking age - Bayesian analysis
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/rd_pymc_drinking.png
:link: rd_pymc_drinking
:link-type: doc
:::
:::{grid-item-card} Sharp regression discontinuity with scikit-learn models
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/rd_skl.png
:link: rd_skl
:link-type: doc
:::
:::{grid-item-card} Drinking age with a scikit-learn model
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/rd_skl_drinking.png
:link: rd_skl_drinking
:link-type: doc
:::
::::

## Regression Kink Design

A variation of regression discontinuity where treatment affects the slope (rate of change) of the outcome with respect to the running variable, rather than causing a discrete jump. The treatment effect is identified by a change in the slope at the cutoff point.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Regression kink design with `pymc` models
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/rkink_pymc.png
:link: rkink_pymc
:link-type: doc
:::
::::

## Synthetic Control

The synthetic control method is a statistical method used to evaluate the effect of an intervention in comparative case studies. It involves the construction of a weighted combination of groups used as controls, to which the treatment group is compared.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Synthetic control with `pymc` models
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/sc_pymc.png
:link: sc_pymc
:link-type: doc
:::
:::{grid-item-card} The effects of Brexit
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/sc_pymc_brexit.png
:link: sc_pymc_brexit
:link-type: doc
:::
:::{grid-item-card} Synthetic control with scikit-learn models
:class-card: sd-card-h-100
:img-top: ../_static/thumbnails/sc_skl.png
:link: sc_skl
:link-type: doc
:::
::::
