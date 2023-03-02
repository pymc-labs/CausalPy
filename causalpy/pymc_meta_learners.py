import pandas as pd
import numpy as np
import pymc as pm


from causalpy.skl_meta_learners import MetaLearner, SLearner, TLearner, XLearner, DRLearner


class BayesianMetaLearner(MetaLearner):
    "Base class for PyMC based meta-learners."

    def __init__(self, X: pd.DataFrame, y: pd.Series, treated: pd.Series) -> None:
        super().__init__(X, y, treated)


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

    def _compute_cate(self, X):
        cate_t = self.models["treated_cate"].predict(X)["posterior_predictive"].mu
        cate_u = self.models["treated_cate"].predict(X)["posterior_predictive"].mu
        g = self.models["propensity"].predict(X)["posterior_predictive"].mu
        return g * cate_u + (1 - g) * cate_t

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        # TODO
        pass


class BayesianDRLearner(DRLearner, BayesianMetaLearner):
    "PyMC version of DR-Learner."

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        # TODO
        pass
