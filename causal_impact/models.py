import pymc as pm
import numpy as np
import pandas as pd
from causal_impact.causal_model_builder import CausalModelBuilder


class SyntheticControl(CausalModelBuilder):
    _model_type = "SyntheticControl"
    version = "0.1"

    def build_model(self, model_config, data=None):
        if data is not None:
            X = pm.MutableData("X", data[model_config["predictor_vars"]].to_numpy())
            y = pm.MutableData("y", data[model_config["target_var"]].to_numpy())

        n_predictors = len(model_config["predictor_vars"])
        # # prior parameters
        # a_loc = model_config["a_loc"]
        # a_scale = model_config["a_scale"]
        # b_loc = model_config["b_loc"]
        # b_scale = model_config["b_scale"]
        # obs_error = model_config["obs_error"]

        # priors
        beta = pm.Dirichlet("beta", a=np.ones(n_predictors))
        sigma = pm.HalfNormal("sigma", 1)

        # model
        mu = pm.Deterministic("mu", pm.math.dot(X, beta))

        # observed data
        if data is not None:
            pm.Normal("y_model", mu, sigma, shape=y.shape, observed=y)

    def _data_setter(self, data: pd.DataFrame):
        """Set the data for the post-treatment period"""
        with self.model:
            pm.set_data(
                {
                    "X": data[self.model_config["predictor_vars"]].to_numpy(),
                    "y": data[self.model_config["target_var"]].to_numpy(),
                }
            )

    # @classmethod
    # def create_sample_input(cls):
    #     x = np.linspace(start=0, stop=70, num=100)
    #     y = 5 * x + 3
    #     y = y + np.random.normal(0, 1, len(x))
    #     data = pd.DataFrame({"input": x, "output": y})

    #     model_config = {
    #         "a_loc": 0,
    #         "a_scale": 10,
    #         "b_loc": 0,
    #         "b_scale": 10,
    #         "obs_error": 2,
    #     }

    #     sampler_config = {
    #         "draws": 1_000,
    #         "tune": 1_000,
    #         "chains": 3,
    #         "target_accept": 0.95,
    #     }

    #     return data, model_config, sampler_config
