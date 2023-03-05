import pandas as pd
import numpy as np
import pymc as pm
import xarray as xa

from causalpy.utils import _fit
from causalpy.skl_meta_learners import MetaLearner, SLearner, TLearner, XLearner, DRLearner
from causalpy.pymc_models import LogisticRegression

class BayesianMetaLearner(MetaLearner):
    "Base class for PyMC based meta-learners."

    def fit(self, X: pd.DataFrame, y: pd.Series, treated: pd.Series, coords=None):
        "Fits model."
        raise NotImplementedError()
    



class BayesianSLearner(SLearner, BayesianMetaLearner):
    "PyMC version of S-Learner."

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        X_untreated = X.assign(treatment=0)
        X_treated = X.assign(treatment=1)
        m = self.models["model"]

        pred_treated = m.predict(X_treated)["posterior_predictive"].mu
        pred_untreated = m.predict(X_untreated)["posterior_predictive"].mu

        cate = pred_treated - pred_untreated

        return cate


class BayesianTLearner(TLearner, BayesianMetaLearner):
    "PyMC version of T-Learner."

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        treated_model = self.models["treated"]
        untreated_model = self.models["untreated"]

        pred_treated = treated_model.predict(X)["posterior_predictive"].mu
        pred_untreated = untreated_model.predict(X)["posterior_predictive"].mu

        cate = pred_treated - pred_untreated

        return cate


class BayesianXLearner(XLearner, BayesianMetaLearner):
    "PyMC version of X-Learner."

    def __init__(
        self,
        X,
        y,
        treated,
        model=None,
        treated_model=None,
        untreated_model=None,
        treated_cate_estimator=None,
        untreated_cate_estimator=None,
        propensity_score_model=LogisticRegression()
    ) -> None:

        super().__init__(
            X,
            y,
            treated,
            model,
            treated_model,
            untreated_model,
            treated_cate_estimator,
            untreated_cate_estimator,
            propensity_score_model
            )
        
    def fit(self, X: pd.DataFrame, y: pd.Series, treated: pd.Series, coords=None):
        (
            treated_model,
            untreated_model,
            treated_cate_estimator,
            untreated_cate_estimator,
            propensity_score_model
        ) = self.models.values()

        # Split data to treated and untreated subsets
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]

        # Estimate response function
        _fit(treated_model, X_t, y_t, coords)
        _fit(untreated_model, X_u, y_u, coords)

        pred_u_t = (
            untreated_model
            .predict(X_t)["posterior_predictive"]
            .mu
            .mean(dim=["chain", "draw"])
            .to_numpy()
            )
        pred_t_u = (
            treated_model
            .predict(X_u)["posterior_predictive"]
            .mu
            .mean(dim=["chain", "draw"])
            .to_numpy()
            )

        tau_t = y_t - pred_u_t
        tau_u = y_u - pred_t_u

        # Estimate CATE separately on treated and untreated subsets
        _fit(treated_cate_estimator, X_t, tau_t, coords)
        _fit(untreated_cate_estimator, X_u, tau_u, coords)

        # Fit propensity score model
        _fit(propensity_score_model, X, treated, coords)
        return self


    def _compute_cate(self, X):
        cate_t = self.models["treated_cate"].predict(X)["posterior_predictive"].mu
        cate_u = self.models["treated_cate"].predict(X)["posterior_predictive"].mu
        g = self.models["propensity"].predict(X)["posterior_predictive"].mu
        return g * cate_u + (1 - g) * cate_t

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        treated_model = self.models["treated_cate"]
        untreated_model =  self.models["untreated_cate"]

        cate_estimate_treated = treated_model.predict(X)["posterior_predictive"].mu
        cate_estimate_untreated = untreated_model.predict(X)["posterior_predictive"].mu
        g = self.models["propensity"].predict(X)["posterior_predictive"].mu

        return g * cate_estimate_untreated + (1 - g) * cate_estimate_treated


class BayesianDRLearner(DRLearner, BayesianMetaLearner):
    "PyMC version of DR-Learner."

    def __init__(
        self,
        X,
        y,
        treated,
        model=None,
        treated_model=None,
        untreated_model=None,
        propensity_score_model=LogisticRegression()
    ):
        super().__init__(X, y, treated, model, treated_model, untreated_model, propensity_score_model)

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        m1 = self.models["treated"].predict(X)
        m0 = self.models["untreated"].predict(X)
        return m1 - m0
    
    def _compute_cate(self, X, y, treated):
        g = self.models["propensity"].predict(X)["posterior_predictive"].mu
        m0 = self.models["untreated"].predict(X)["posterior_predictive"].mu
        m1 = self.models["treated"].predict(X)["posterior_predictive"].mu


        # Broadcast target and treated variables to the size of the predictions
        y0 = xa.DataArray(y, dims="obs_ind")
        y0 = xa.broadcast(y0, m0)[0]

        t0 = xa.DataArray(treated, dims="obs_ind")
        t0 = xa.broadcast(t0, m0)[0]

        cate = (t0 * (y0 - m1) / g + m1 - ((1 - t0) * (y0 - m0) / (1 - g) + m0))

        return cate