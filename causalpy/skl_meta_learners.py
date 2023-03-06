import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_consistent_length

from causalpy.utils import _is_variable_dummy_coded, _fit


class MetaLearner:
    "Base class for meta-learners."

    def __init__(self, X: pd.DataFrame, y: pd.Series, treated: pd.Series) -> None:
        # Check whether input is appropriate
        check_consistent_length(X, y, treated)
        if not _is_variable_dummy_coded(treated):
            raise ValueError("Treatment variable is not dummy coded.")

        self.cate = None
        self.treated = treated
        self.X = X
        self.y = y
        self.models = {}

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        """
        Predict out-of-sample conditional average treatment effect on given input X.
        For in-sample treatement effect self.cate should be used.
        """
        raise NotImplementedError()

    def predict_ate(self, X: pd.DataFrame) -> np.float64:
        """
        Predict out-of-sample average treatment effect on given input X. For in-sample treatement
        effect self.ate() should be used.
        """
        return self.predict_cate(X).mean()

    def ate(self):
        "Returns in-sample average treatement effect."
        return self.cate.mean()

    def fit(self, X: pd.DataFrame, y: pd.Series, treated: pd.Series, coords=None):
        "Fits model."
        raise NotImplementedError()

    def summary(self):
        "Prints summary."
        print(f"Number of observations:            {self.X.shape[0]}")
        print(f"Number of treated observations:    {self.treated.sum()}")
        print(f"Average treatement effect (ATE):   {self.predict_ate(self.X.shape[0])}")

    def plot(self):
        "Plots results. Content is undecided yet."
        plt.hist(self.cate, bins=40, density=True)


class SkMetaLearner(MetaLearner):
    "Base class for sklearn based meta-learners."

    def bootstrap(
        self,
        X_ins: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        X: pd.DataFrame = None,
        n_iter: int = 1000
    ) -> np.array:
        """
        Runs bootstrap n_iter times on a sample of size n_samples.
        Fits on (X_ins, y, treated), then predicts on X.
        """
        if X is None:
            X = X_ins

        # Bootstraping overwrites these attributes
        models, cate = self.models, self.cate

        # Calculate number of treated and untreated data points
        n1 = self.treated.sum()
        n0 = self.treated.count() - n1

        # Prescribed treatement variable of samples
        t_bs = pd.Series(n0 * [0] + n1 * [1], name="treatement")

        results = []

        # TODO: paralellize this loop
        for _ in range(n_iter):
            # Take sample with replacement from our data in a way that we have
            # the same number of treated and untreated data points as in the whole
            # data set.
            X0_bs = X_ins[treated == 0].sample(n=n0, replace=True)
            y0_bs = y.loc[X0_bs.index]

            X1_bs = X_ins[treated == 1].sample(n=n1, replace=True)
            y1_bs = y.loc[X1_bs.index]

            X_bs = pd.concat([X0_bs, X1_bs], axis=0).reset_index(drop=True)
            y_bs = pd.concat([y0_bs, y1_bs], axis=0).reset_index(drop=True)

            self.fit(X_bs.reset_index(drop=True), y_bs, t_bs)
            results.append(self.predict_cate(X))

        self.models = models
        self.cate = cate
        return np.array(results)

    def ate_confidence_interval(
        self,
        X_ins: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        X: pd.DataFrame = None,
        q: float = .05,
        n_iter: int = 1000
    ) -> tuple:
        "Estimates confidence intervals for ATE on X using bootstraping."
        cates = self.bootstrap(X_ins, y, treated, X, n_iter)
        ates = cates.mean(axis=0)
        return np.quantile(ates, q=q / 2), np.quantile(cates, q=1 - q / 2)

    def cate_confidence_interval(
        self,
        X_ins: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        X: pd.DataFrame = None,
        q: float = .05,
        n_iter: int = 1000
    ) -> np.array:
        "Estimates confidence intervals for CATE on X using bootstraping."
        cates = self.bootstrap(X_ins, y, treated, X, n_iter)
        conf_ints = np.append(
            np.quantile(cates, q / 2, axis=0).reshape(-1, 1),
            np.quantile(cates, 1 - q / 2, axis=0).reshape(-1, 1),
            axis=1,
        )
        return conf_ints

    def bias(
        self,
        X_ins: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        X: pd.DataFrame = None,
        q: float = .05,
        n_iter: int = 1000
    ):
        cates = self.bootstrap(X_ins, y, treated, X, n_iter)

    def summary(self, n_iter=1000):
        conf_ints = self.ate_confidence_interval(
            self.X, self.y, self.treated, self.X, n_iter=n_iter)
        print(f"Number of observations:            {self.X.shape[0]}")
        print(f"Number of treated observations:    {self.treated.sum()}")
        print(f"Average treatement effect (ATE):   {self.predict_ate(self.X)}")
        print(f"Confidence interval for ATE:       {conf_ints}")


class SLearner(SkMetaLearner):
    """
    Implements of S-learner described in [1]. S-learner estimates conditional average
    treatment effect with the use of a single model.

    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu.
        Metalearners for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    """

    def __init__(
        self, X: pd.DataFrame, y: pd.Series, treated: pd.Series, model
    ) -> None:
        super().__init__(X=X, y=y, treated=treated)
        self.models["model"] = model

        COORDS = {
            "coeffs": list(X.columns) + ["treated"],
            "obs_indx": np.arange(X.shape[0])
        }

        self.fit(X, y, treated, coords=COORDS)
        self.cate = self.predict_cate(X)

    def fit(self, X: pd.DataFrame, y: pd.Series, treated: pd.Series, coords=None):
        X_t = X.assign(treatment=treated)
        _fit(model=self.models["model"], X=X_t, y=y, coords=coords)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        X_control = X.assign(treatment=0)
        X_treated = X.assign(treatment=1)
        m = self.models["model"]
        return m.predict(X_treated) - m.predict(X_control)


class TLearner(SkMetaLearner):
    """
    Implements of T-learner described in [1]. T-learner fits two separate models to estimate
    conditional average treatment effect.

    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu.
        Metalearners for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        model=None,
        treated_model=None,
        untreated_model=None
    ) -> None:
        super().__init__(X=X, y=y, treated=treated)

        if model is None and (untreated_model is None or treated_model is None):
            raise ValueError(
                "Either model or both of treated_model and untreated_model \
                have to be specified."
            )
        elif not (model is None or untreated_model is None or treated_model is None):
            raise ValueError(
                "Either model or both of treated_model and untreated_model \
                have to be specified."
            )

        if model is not None:
            untreated_model = deepcopy(model)
            treated_model = deepcopy(model)

        self.models = {"treated": treated_model, "untreated": untreated_model}

        COORDS = {"coeffs": X.columns, "obs_indx": np.arange(X.shape[0])}

        self.fit(X, y, treated, coords=COORDS)
        self.cate = self.predict_cate(X)

    def fit(self, X: pd.DataFrame, y: pd.Series, treated: pd.Series, coords=None):
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]
        _fit(model=self.models["treated"], X=X_t, y=y_t, coords=coords)
        _fit(model=self.models["untreated"], X=X_u, y=y_u, coords=coords)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        treated_model = self.models["treated"]
        untreated_model = self.models["untreated"]
        return treated_model.predict(X) - untreated_model.predict(X)


class XLearner(SkMetaLearner):
    """
    Implements of X-learner introduced in [1]. X-learner estimates conditional average treatment
    effect with the use of five separate models.

    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu.
        Metalearners for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    """

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
        propensity_score_model=LogisticRegression(penalty=None)
    ):
        super().__init__(X=X, y=y, treated=treated)

        if model is None and (untreated_model is None or treated_model is None):
            raise ValueError(
                "Either model or each of treated_model, untreated_model, \
                treated_cate_estimator, untreated_cate_estimator has to be specified."
            )
        elif not (model is None or untreated_model is None or treated_model is None):
            raise ValueError(
                "Either model or each of treated_model, untreated_model, \
                treated_cate_estimator, untreated_cate_estimator has to be specified."
            )

        if model is not None:
            treated_model = deepcopy(model)
            untreated_model = deepcopy(model)
            treated_cate_estimator = deepcopy(model)
            untreated_cate_estimator = deepcopy(model)

        self.models = {
            "treated": treated_model,
            "untreated": untreated_model,
            "treated_cate": treated_cate_estimator,
            "untreated_cate": untreated_cate_estimator,
            "propensity": propensity_score_model
        }

        COORDS = {"coeffs": X.columns, "obs_indx": np.arange(X.shape[0])}

        self.fit(X, y, treated, coords=COORDS)

        # Compute cate
        self.cate = self._compute_cate(X)

    def _compute_cate(self, X):
        "Computes cate for given input."
        cate_t = self.models["treated_cate"].predict(X)
        cate_u = self.models["treated_cate"].predict(X)
        g = self.models["propensity"].predict_proba(X)[:, 1]
        return g * cate_u + (1 - g) * cate_t

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

        tau_t = y_t - untreated_model.predict(X_t)
        tau_u = treated_model.predict(X_u) - y_u

        # Estimate CATE separately on treated and untreated subsets
        _fit(treated_cate_estimator, X_t, tau_t, coords)
        _fit(untreated_cate_estimator, X_u, tau_u, coords)

        # Fit propensity score model
        _fit(propensity_score_model, X, treated, coords)
        return self

    def predict_cate(self, X):
        cate_estimate_treated = self.models["treated_cate"].predict(X)
        cate_estimate_untreated = self.models["untreated_cate"].predict(X)

        g = self.models["propensity"].predict_proba(X)[:, 1]

        return g * cate_estimate_untreated + (1 - g) * cate_estimate_treated


class DRLearner(SkMetaLearner):
    """
    Implements of DR-learner also known as doubly robust learner as described in [1].

    [1] Curth, Alicia, Mihaela van der Schaar.
        Nonparametric estimation of heterogeneous treatment effects: From theory to learning algorithms.
        International Conference on Artificial Intelligence and Statistics, pp. 1810-1818 (2021).

    """

    def __init__(
        self,
        X,
        y,
        treated,
        model=None,
        treated_model=None,
        untreated_model=None,
        propensity_score_model=LogisticRegression(penalty=None)
    ):
        super().__init__(X=X, y=y, treated=treated)

        if model is None and (untreated_model is None or treated_model is None):
            raise ValueError(
                "Either model or each of treated_model, untreated_model, \
                treated_cate_estimator, untreated_cate_estimator has to be specified."
            )
        elif not (model is None or untreated_model is None or treated_model is None):
            raise ValueError(
                "Either model or each of treated_model, untreated_model, \
                treated_cate_estimator, untreated_cate_estimator has to be specified."
            )

        if model is not None:
            treated_model = deepcopy(model)
            untreated_model = deepcopy(model)

        # Estimate response function
        self.models = {
            "treated": treated_model,
            "untreated": untreated_model,
            "propensity": propensity_score_model
        }

        COORDS = {"coeffs": X.columns, "obs_indx": np.arange(X.shape[0])}
        self.fit(X, y, treated, coords=COORDS)

        # Estimate CATE
        self.cate = self._compute_cate(X, y, treated)

    def _compute_cate(self, X, y, treated):
        g = self.models["propensity"].predict_proba(X)[:, 1]
        m0 = self.models["untreated"].predict(X)
        m1 = self.models["treated"].predict(X)

        cate = (treated * (y - m1) / g + m1
                - ((1 - treated) * (y - m0) / (1 - g) + m0))

        return cate

    def fit(self, X: pd.DataFrame, y: pd.Series, treated: pd.Series, coords=None):
        # Split data to treated and untreated subsets
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]

        treated_model, untreated_model, propensity_score_model = self.models.values()

        # Estimate response functions
        _fit(treated_model, X_t, y_t, coords)
        _fit(untreated_model, X_u, y_u, coords)

        # Fit propensity score model
        _fit(propensity_score_model, X, treated, coords)

        return self

    def predict_cate(self, X):
        m1 = self.models["treated"].predict(X)
        m0 = self.models["untreated"].predict(X)
        return m1 - m0
