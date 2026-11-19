---
myst:
  html_meta:
    description: "CausalPy is a Python library for Bayesian causal inference in quasi-experiments: synthetic control, difference-in-differences, interrupted time series, regression discontinuity, instrumental variables, and inverse propensity score weighting."
"og:description": "CausalPy is a Python library for Bayesian causal inference in quasi-experiments: synthetic control, difference-in-differences, interrupted time series, regression discontinuity, instrumental variables, and inverse propensity score weighting."
---

:::{image} _static/logo.png
:width: 40 %
:align: center
:alt: CausalPy logo
:::

# CausalPy

CausalPy helps you estimate what changed because of an intervention when a randomized experiment is not available. Use Bayesian models or scikit-learn with methods such as interrupted time series, synthetic control, difference-in-differences and regression discontinuity. <a class="cp-methods-pill" href="index.html#methods-and-workflows">More methods →</a>

<p class="cp-example-label">Example: Interrupted time series</p>

<iframe class="cp-animation-frame" src="_static/animations/interrupted-time-series.html" title="Interrupted time series: fit, forecast and estimate impact" loading="lazy" height="740"></iframe>

<details class="cp-animation-details">
<summary>About this example &amp; the calculation</summary>

The animation uses synthetic monthly outcomes with a trend, annual seasonality, independent noise, and a known +18-unit intervention from January 2024. CausalPy fits the first 48 months and forecasts the next 24 to estimate a {term}`counterfactual`. All fitted values and impacts are computed by CausalPy and saved for playback; the animation timing does not represent fitting time.

White points show the observations. Green bands show pointwise 94% highest-density credible intervals for the expected untreated outcome. Green bars show observed outcomes minus the expected counterfactual; white error bars show their corresponding 94% credible intervals. The running average and its numeric 94% interval summarize the post-intervention months revealed so far, using the same pre-intervention fit. These intervals include uncertainty in the fitted mean, but exclude new observation noise and uncertainty about the causal assumptions.

A causal interpretation requires that the untreated pattern would have continued and that no concurrent event explains the difference. A good fit before the intervention does not establish these assumptions.

```python
experiment = cp.InterruptedTimeSeries(
    df, pd.Timestamp("2024-01-01"),
    formula="y ~ 1 + t + season",
    model=cp.pymc_models.LinearRegression(),
).fit()
```

See the {doc}`worked Python example <notebooks/interrupted-time-series-pymc>` for a complete interrupted time series analysis.

</details>

:::::{card} Have a causal inference project in mind?
:class-card: pm-project

Work with **PyMC Labs, the team behind CausalPy**, on lift and geo tests, {term}`quasi-experiments <Quasi-experiment>`, and high-stakes causal analysis. These calls are for consulting inquiries only. For technical usage questions and free community support, use [GitHub Discussions](https://github.com/pymc-labs/CausalPy/discussions/categories/q-a).

::::{div} pm-project-actions

:::{button-ref} project-brief
:color: primary
:class: pm-cta-button

Submit an RFP
:::

:::{button-link} https://calendly.com/benjamin-vincent/causalpy
:color: primary
:class: pm-cta-button

Book a call
:::

:::{button-link} https://www.pymc-labs.com/contact
:color: primary
:class: pm-cta-button

Contact PyMC Labs
:::
::::
:::::

```{include} ../../README.md
:start-after: <!-- docs-cta-end -->
```

::::{toctree}
:hidden:

knowledgebase/index
api/index
notebooks/index
::::
