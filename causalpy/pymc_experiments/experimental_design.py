"""
experimental_design.py

Provides ExperimentalDesign base class used by most experiments.
"""

import arviz as az


class ExperimentalDesign:
    """
    Base class for other experiment types

    See subclasses for examples of most methods
    """

    model = None
    expt_type = None

    def __init__(self, model=None, **kwargs):
        if model is not None:
            self.model = model
        if self.model is None:
            raise ValueError("fitting_model not set or passed.")

    @property
    def idata(self):
        """
        Access to the models InferenceData object
        """

        return self.model.idata

    def print_coefficients(self) -> None:
        """
        Prints the model coefficients

        Example
        --------
        >>> import causalpy as cp
        >>> df = cp.load_data("did")
        >>> seed = 42
        >>> result = cp.pymc_experiments.DifferenceInDifferences(
        ...     df,
        ...     formula="y ~ 1 + group*post_treatment",
        ...     time_variable_name="t",
        ...     group_variable_name="group",
        ...     model=cp.pymc_models.LinearRegression(
        ...             sample_kwargs={
        ...                 "draws": 2000,
        ...                 "random_seed": seed,
        ...                 "progressbar": False
        ...             }),
        ...  )
        >>> result.print_coefficients() # doctest: +NUMBER
        Model coefficients:
        Intercept                     1.0, 94% HDI [1.0, 1.1]
        post_treatment[T.True]        0.9, 94% HDI [0.9, 1.0]
        group                         0.1, 94% HDI [0.0, 0.2]
        group:post_treatment[T.True]  0.5, 94% HDI [0.4, 0.6]
        sigma                         0.0, 94% HDI [0.0, 0.1]
        """
        print("Model coefficients:")
        coeffs = az.extract(self.idata.posterior, var_names="beta")
        # Note: f"{name: <30}" pads the name with spaces so that we have alignment of
        # the stats despite variable names of different lengths
        for name in self.labels:
            coeff_samples = coeffs.sel(coeffs=name)
            print(
                f"{name: <30}{coeff_samples.mean().data:.2f}, 94% HDI [{coeff_samples.quantile(0.03).data:.2f}, {coeff_samples.quantile(1-0.03).data:.2f}]"  # noqa: E501
            )
        # add coeff for measurement std
        coeff_samples = az.extract(self.model.idata.posterior, var_names="sigma")
        name = "sigma"
        print(
            f"{name: <30}{coeff_samples.mean().data:.2f}, 94% HDI [{coeff_samples.quantile(0.03).data:.2f}, {coeff_samples.quantile(1-0.03).data:.2f}]"  # noqa: E501
        )
