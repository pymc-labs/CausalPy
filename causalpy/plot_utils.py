import arviz as az


def plot_xY(x, Y, ax, plot_hdi_kwargs=dict()):
    quantiles = Y.quantile(
        (0.025, 0.25, 0.5, 0.75, 0.975), dim=("chain", "draw")
    ).transpose()

    az.plot_hdi(
        x,
        hdi_data=quantiles.sel(quantile=[0.025, 0.975]),
        fill_kwargs={"alpha": 0.25},
        smooth=False,
        ax=ax,
        **plot_hdi_kwargs,
    )
    az.plot_hdi(
        x,
        hdi_data=quantiles.sel(quantile=[0.25, 0.75]),
        fill_kwargs={"alpha": 0.5},
        smooth=False,
        ax=ax,
        **plot_hdi_kwargs,
    )
    ax.plot(x, quantiles.sel(quantile=0.5), color="k")
