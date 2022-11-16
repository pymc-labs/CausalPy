import arviz as az


def plot_xY(
    x, Y, ax, plot_hdi_kwargs=dict(), hdi_prob: float = 0.94, include_label: bool = True
) -> None:
    """Utility function to plot HDI intervals."""

    Y = Y.stack(samples=["chain", "draw"]).T
    az.plot_hdi(
        x,
        Y,
        hdi_prob=hdi_prob,
        fill_kwargs={
            "alpha": 0.25,
            "label": f"{hdi_prob*100}% HDI" if include_label else None,
        },
        smooth=False,
        ax=ax,
        **plot_hdi_kwargs,
    )
    ax.plot(
        x,
        Y.mean(dim="samples"),
        color="k",
        label="Posterior mean" if include_label else None,
    )
