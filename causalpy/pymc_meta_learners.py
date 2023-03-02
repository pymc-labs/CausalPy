import pandas as pd
import numpy as np
import pymc as pm


from causalpy.skl_meta_learners import SLearner, TLearner, XLearner, DRLearner


class BayesianMetaLearner:
    "Base class for PyMC based meta-learners."

    def plot(self):
        # TODO
        pass

    def summary(self):
        # TODO
        pass


class BayesianSLearner(BayesianMetaLearner, SLearner):
    "PyMC version of S-Learner."

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        X_untreated = X.assign(treatment=0)
        X_treated = X.assign(treatment=1)
        m = self.models['model']

        pred_treated = m.predict(X_treated)["posterior_predictive"].mu
        pred_untreated = m.predict(X_untreated)["posterior_predictive"].mu

        self.cate_posterior = pred_treated - pred_untreated

        return self.cate_posterior.mean(dim=["chain", "draw"])


class BayesianTLearner(BayesianMetaLearner, TLearner):
    "PyMC version of T-Learner."

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        treated_model = self.models["treated"]
        untreated_model = self.models["untreated"]

        pred_treated = treated_model.predict(X)["posterior_predictive"].mu
        pred_untreated = untreated_model.predict(X)["posterior_predictive"].mu

        self.cate_posterior = pred_treated - pred_untreated

        return self.cate_posterior.mean(dim=["chain", "draw"])


class BayesianXLearner(BayesianMetaLearner, XLearner):
    "PyMC version of X-Learner."

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        # TODO
        pass


class BayesianDRLearner(BayesianMetaLearner, DRLearner):
    "PyMC version of DR-Learner."

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        # TODO
        pass
