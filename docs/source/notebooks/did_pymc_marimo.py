import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Difference in Differences with `pymc` models

    :::{note}
    This example is in-progress! Further elaboration and explanation will follow soon.
    :::
    """)
    return


@app.cell
def _():
    import arviz as az

    import causalpy as cp

    return az, cp


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'retina'
    seed = 42
    return (seed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(cp):
    df = cp.load_data("did")
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the analysis

    :::{note}
    The `random_seed` keyword argument for the PyMC sampler is not necessary. We use it here so that the results are reproducible.
    :::
    """)
    return


@app.cell
def _(cp, df, seed):
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(sample_kwargs={"random_seed": seed}),
    )
    return (result,)


@app.cell
def _(result):
    fig, _ax = result.plot()
    return


@app.cell
def _(result):
    result.summary()
    return


@app.cell
def _(az, result):
    _ax = az.plot_posterior(result.causal_impact, ref_val=0)
    _ax.set(title="Posterior estimate of causal impact")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect Summary Reporting

    For decision-making, you often need a concise summary of the causal effect. The `effect_summary()` method provides a decision-ready report with key statistics. Note that for Difference-in-Differences, the effect is a single scalar (average treatment effect), unlike time-series experiments where effects vary over time.
    """)
    return


@app.cell
def _(result):
    # Generate effect summary
    stats = result.effect_summary()
    stats.table
    return (stats,)


@app.cell
def _(stats):
    print(stats.text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can customize the summary with different directions and ROPE thresholds:

    - **Direction**: Test for increase, decrease, or two-sided effect
    - **Alpha**: Set the HDI confidence level (default 95%)
    - **ROPE**: Specify a minimal effect size threshold
    """)
    return


@app.cell
def _(result):
    # Example: Two-sided test with ROPE
    stats_1 = result.effect_summary(direction="two-sided", alpha=0.05, min_effect=0.3)
    stats_1.table  # Region of Practical Equivalence
    return (stats_1,)


@app.cell
def _(stats_1):
    print("\n" + stats_1.text)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
