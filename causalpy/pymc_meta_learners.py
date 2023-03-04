import pandas as pd
import numpy as np
import pymc as pm


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