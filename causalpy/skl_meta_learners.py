"Scikit-learn based meta-learners."
from copy import deepcopy
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import check_consistent_length

from causalpy.summary import Summary
from causalpy.utils import _fit, _is_variable_dummy_coded


class MetaLearner:
    "Base class for meta-learners."

    def __init__(self, X: pd.DataFrame, y: pd.Series, treated: pd.Series) -> None:
        # Check whether input is appropriate
        check_consistent_length(X, y, treated)
        if not _is_variable_dummy_coded(treated):
            raise ValueError("Treatment variable is not dummy coded.")

        # Check whether input data share the same index
        if not ((X.index == y.index) & (y.index == treated.index)).all():
            raise ValueError("Indices of input data do not coincide.")

        self.cate = None
        self.treated = treated
        self.X = X
        self.y = y
        self.models = {}
        self.labels = X.columns
        self.index = X.index

    def predict_cate(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict out-of-sample conditional average treatment effect on given input X.
        For in-sample treatement effect self.cate should be used.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_featues).
        """
        raise NotImplementedError()

    def predict_ate(self, X: pd.DataFrame) -> np.float64:
        """
        Predict out-of-sample average treatment effect on given input X. For in-sample
        treatement effect self.ate() should be used.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_featues).
        """
        return self.predict_cate(X).mean()

    def ate(self) -> np.float64:
        "Returns in-sample average treatement effect."
        return self.cate.mean()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        coords: Dict[str, Any] = None,
    ):
        """
        Fits model.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_featues).
            Training data.
        y : pandas.Series of shape (n_samples, ).
            Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.
        coords : Dict[str, Any].
            Dictionary containing the keys coeffs and obs_indx.
        """
        raise NotImplementedError()

    def plot(self):
        "Plots results. Content is undecided yet."
        plt.hist(self.cate, bins=40, density=True)


class SkMetaLearner(MetaLearner):
    "Base class for sklearn based meta-learners."

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        coords: Dict[str, Any] = None,
    ):
        raise NotImplementedError()

    def bootstrap(
        self,
        X_ins: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        X: pd.DataFrame = None,
        n_iter: int = 1000,
    ) -> np.array:
        """
        Runs bootstrap n_iter times on a sample of size n_samples.
        Fits on (X_ins, y, treated), then predicts on X.

        Parameters
        ----------
        X_ins : pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y : pandas.Series of shape (n_samples, ).
            Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        n_iter : int, default = 1000.
            Number of bootstrap iterations to perform.
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
        q: float = 0.05,
        n_iter: int = 1000,
    ) -> tuple:
        """Estimates confidence intervals for ATE on X using bootstraping.

        Parameters
        ----------
        X_ins : pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y : pandas.Series of shape (n_samples, ).
            Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        q : float, default=.05.
            Quantile to compute. Should be between in the interval [0, 1].
        n_iter : int, default = 1000.
            Number of bootstrap iterations to perform.
        """
        cates = self.bootstrap(X_ins, y, treated, X, n_iter)
        ates = cates.mean(axis=0)
        return np.quantile(ates, q=q / 2), np.quantile(cates, q=1 - q / 2)

    def cate_confidence_interval(
        self,
        X_ins: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        X: pd.DataFrame = None,
        q: float = 0.05,
        n_iter: int = 1000,
    ) -> np.array:
        """Estimates confidence intervals for CATE on X using bootstraping.

        Parameters
        ----------
        X_ins : pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y : pandas.Series of shape (n_samples, ).
            Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        q : float, default=.05.
            Quantile to compute. Should be between in the interval [0, 1].
        n_iter : int, default = 1000.
            Number of bootstrap iterations to perform."""
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
        n_iter: int = 1000,
    ) -> np.float64:
        """Calculates bootstrap estimate of bias of CATE estimator.

        Parameters
        ----------
        X_ins : pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y : pandas.Series of shape (n_samples, ).
            Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        q : float, default=.05.
            Quantile to compute. Should be between in the interval [0, 1].
        n_iter : int, default = 1000.
            Number of bootstrap iterations to perform."""
        if X is None:
            X = X_ins

        pred = self.predict_cate(X=X)
        bs_pred = self.bootstrap(X_ins, y, treated, X, n_iter).mean(axis=0)

        return (bs_pred - pred).mean()

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ) -> Dict[str, np.float64]:
        """
        Returns a dictionary of R^2 scores of base-learners, mean accuracy in case of
        propensity score estimator.

        Parameters
        ----------
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        y : pandas.Series of shape (n_samples, ).
            Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.
        """
        raise NotImplementedError()
    
    def average_uplift_by_percentile(
            self,
            X: pd.DataFrame,
            nbins: int=20,
            ) -> pd.DataFrame:
        """
        Returns average uplift in quantile groups. 

        Parameters
        ----------
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        nbins : int.
            Number of bins.
        """
        preds = self.predict_cate(X)
        nunique = preds.unique().shape[0]

        df = pd.DataFrame(
            {
            "Mean uplift by quantile": preds,
            "quantile": pd.qcut(preds, q=min(nunique, nbins))
            }
            ).groupby("quantile").mean()
        
        return df

        
    def summary(self, n_iter: int = 100) -> Summary:
        """
        Returns.

        Parameters
        ----------
        n_iter :    int, default=1000.
                    Number of bootstrap iterations to perform.
        """
        # TODO: we run self.bootstrap twice independently.
        conf_ints = self.ate_confidence_interval(
            self.X, self.y, self.treated, self.X, n_iter=n_iter,
        )
        conf_ints = map(lambda x: round(x, 2), conf_ints)
        bias = self.bias(self.X, self.y, self.treated, self.X, n_iter=n_iter)
        bias = round(bias, 2)
        ate = round(self.ate(), 2)
        score = self.score(self.X, self.y, self.treated)
        models = {k: [type(v).__name__, round(score[k], 2)] for k, v in self.models.items()}

        s = Summary()
        s.add_title(["Conditional Average Treatment Effect Estimator Summary"])
        s.add_row("Number of observations", [self.index.shape[0]], 2)
        s.add_row("Number of treated observations", [self.treated.sum()], 2)
        s.add_row("Average treatement effect (ATE)", [ate], 2)
        s.add_row("95% Confidence interval for ATE", [tuple(conf_ints)], 1)
        s.add_row("Estimated bias", [bias], 2)
        s.add_title(["Base learners"])
        s.add_header(["", "Model", "R^2"], 1)

        for name, x in models.items():
            s.add_row(name, x, 1)
        
        return s

class SLearner(SkMetaLearner):
    """
    Implements of S-learner described in [1]. S-learner estimates conditional average
    treatment effect with the use of a single model.

    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners
        for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    Parameters
    ----------
    X :     pandas.DataFrame of shape (n_samples, n_featues).
            Training data.
    y :     pandas.Series of shape (n_samples, ).
            Target vector.
    treated : pandas.Series of shape (n_samples, ).
            Treatement assignment indicator consisting of zeros and ones.
    model : sklearn.base.RegressorMixin.
            Base learner.
    """

    def __init__(
        self, X: pd.DataFrame, y: pd.Series, treated: pd.Series, model: RegressorMixin
    ) -> None:
        super().__init__(X=X, y=y, treated=treated)
        self.models["model"] = model

        COORDS = {"coeffs": list(self.labels) + ["treated"], "obs_indx": self.index}

        self.fit(X, y, treated, coords=COORDS)
        self.cate = self.predict_cate(X)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        coords: Dict[str, Any] = None,
    ):
        X_t = X.assign(treatment=treated)
        _fit(model=self.models["model"], X=X_t, y=y, coords=coords)
        return self

    def predict_cate(self, X: pd.DataFrame) -> pd.Series:
        X_control = X.assign(treatment=0)
        X_treated = X.assign(treatment=1)
        m = self.models["model"]
        return pd.Series(m.predict(X_treated) - m.predict(X_control), index=self.index)

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ) -> Dict[str, np.float64]:
        return {"model": self.models["model"].score(X.assign(treatment=treated), y)}


class TLearner(SkMetaLearner):
    """
    Implements of T-learner described in [1]. T-learner fits two separate models to
    estimate conditional average treatment effect.

    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners
        for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    Parameters
    ----------
    X :     pandas.DataFrame of shape (n_samples, n_featues).
            Training data.
    y :     pandas.Series of shape (n_samples, ).
            Target vector.
    treated : pandas.Series of shape (n_samples, ).
            Treatement assignment indicator consisting of zeros and ones.
    model : sklearn.base.RegressorMixin.
            If specified, it will be used both as treated and untreated model. Either
            model or both of treated_model and untreated_model have to be specified.
    treated_model: sklearn.base.RegressorMixin.
            Model used for predicting target vector for treated values.
    untreated_model: sklearn.base.RegressorMixin.
            Model used for predicting target vector for untreated values.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        model: RegressorMixin = None,
        treated_model: RegressorMixin = None,
        untreated_model: RegressorMixin = None,
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

        COORDS = {"coeffs": self.labels, "obs_indx": self.index}

        self.fit(X, y, treated, coords=COORDS)
        self.cate = self.predict_cate(X)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        coords: Dict[str, Any] = None,
    ):
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]
        _fit(model=self.models["treated"], X=X_t, y=y_t, coords=coords)
        _fit(model=self.models["untreated"], X=X_u, y=y_u, coords=coords)
        return self

    def predict_cate(self, X: pd.DataFrame) -> pd.Series:
        treated_model = self.models["treated"]
        untreated_model = self.models["untreated"]
        return pd.Series(
            treated_model.predict(X) - untreated_model.predict(X), index=self.index
        )

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ) -> Dict[str, np.float64]:
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]
        return {
            "treated": self.models["treated"].score(X_t, y_t),
            "untreated": self.models["untreated"].score(X_u, y_u),
        }


class XLearner(SkMetaLearner):
    """
    Implements of X-learner introduced in [1]. X-learner estimates conditional average
    treatment effect with the use of five separate models.

    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners
        for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    Parameters
    ----------
    X :     pandas.DataFrame of shape (n_samples, n_featues).
            Training data.
    y :     pandas.Series of shape (n_samples, ).
            Target vector.
    treated : pandas.Series of shape (n_samples, ).
            Treatement assignment indicator consisting of zeros and ones.
    model : sklearn.base.RegressorMixin.
            If specified, it will be used in all of the subregressions, except for the
            propensity_score_model. Either model or all of treated_model,
            untreated_model, treated_cate_estimator and untreated_cate_estimator have
            to be specified.
    treated_model : sklearn.base.RegressorMixin.
            Model used for predicting target vector for treated values.
    untreated_model :   sklearn.base.RegressorMixin.
            Model used for predicting target vector for untreated values.
    untreated_cate_estimator :  sklearn.base.RegressorMixin
            Model used for CATE estimation on untreated data.
    treated_cate_estimator :    sklearn.base.RegressorMixin
            Model used for CATE estimation on treated data.
    propensity_score_model :    sklearn.base.ClassifierMixin,
                                default = sklearn.linear_model.LogisticRegression().
            Model used for propensity score estimation.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        model: RegressorMixin = None,
        treated_model: RegressorMixin = None,
        untreated_model: RegressorMixin = None,
        treated_cate_estimator: RegressorMixin = None,
        untreated_cate_estimator: RegressorMixin = None,
        propensity_score_model: ClassifierMixin = LogisticRegression(penalty=None),
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
            "propensity": propensity_score_model,
        }

        COORDS = {"coeffs": self.labels, "obs_indx": self.index}

        self.fit(X, y, treated, coords=COORDS)

        # Compute cate
        self.cate = self.predict_cate(X)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        coords: Dict[str, Any] = None,
    ):
        treated_model = self.models["treated"]
        untreated_model = self.models["untreated"]
        treated_cate_estimator = self.models["treated_cate"]
        untreated_cate_estimator = self.models["untreated_cate"]
        propensity_score_model = self.models["propensity"]

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

    def predict_cate(self, X: pd.DataFrame) -> pd.Series:
        cate_estimate_treated = self.models["treated_cate"].predict(X)
        cate_estimate_untreated = self.models["untreated_cate"].predict(X)
        g = self.models["propensity"].predict_proba(X)[:, 1]

        cate = g * cate_estimate_untreated + (1 - g) * cate_estimate_treated
        return pd.Series(cate, index=self.index)

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ) -> Dict[str, np.float64]:
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]

        tau_t = y_t - self.models["untreated"].predict(X_t)
        tau_u = self.models["treated"].predict(X_u) - y_u

        return {
            "treated": self.models["treated"].score(X_t, y_t),
            "untreated": self.models["untreated"].score(X_u, y_u),
            "propensity": self.models["propensity"].score(X, treated),
            "treated_cate": self.models["treated_cate"].score(X_t, tau_t),
            "untreated_cate": self.models["untreated_cate"].score(X_u, tau_u),
        }


class DRLearner(SkMetaLearner):
    """
    Implements of DR-learner also known as doubly robust learner as described in [1].

    [1] Curth, Alicia, Mihaela van der Schaar.
        Nonparametric estimation of heterogeneous treatment effects: From theory to
        learning algorithms. International Conference on Artificial Intelligence and
        Statistics, pp. 1810-1818 (2021).

    Parameters
        ----------
        X :     pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y :     pandas.Series of shape (n_samples, ).
                Target vector.
        treated : pandas.Series of shape (n_samples, ).
                Treatement assignment indicator consisting of zeros and ones.
        model : sklearn.base.RegressorMixin.
                If specified, it will be used in all of the subregressions, except for
                the propensity_score_model. Either model or all of treated_model,
                untreated_model, treated_cate_estimator and untreated_cate_estimator
                have to be specified.
        treated_model : sklearn.base.RegressorMixin.
                Model used for predicting target vector for treated values.
        untreated_model :   sklearn.base.RegressorMixin.
                Model used for predicting target vector for untreated values.
        pseudo_outcome_model :  sklearn.base.RegressorMixin
                Model used for pseudo-outcome estimation.
        propensity_score_model :    sklearn.base.ClassifierMixin,
                                   default = sklearn.linear_model.LogisticRegression().
                Model used for propensity score estimation.
        cross_fitting : bool, default=False.
                If True, performs a cross fitting step.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        model: RegressorMixin = None,
        treated_model: RegressorMixin = None,
        untreated_model: RegressorMixin = None,
        pseudo_outcome_model: RegressorMixin = None,
        propensity_score_model: ClassifierMixin = LogisticRegression(penalty=None),
        cross_fitting: bool = False,
    ):
        super().__init__(X=X, y=y, treated=treated)

        self.cross_fitting = cross_fitting

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
            pseudo_outcome_model = deepcopy(model)

        # Estimate response function
        self.models = {
            "treated": treated_model,
            "untreated": untreated_model,
            "propensity": propensity_score_model,
            "pseudo_outcome": pseudo_outcome_model,
        }

        cross_fitted_models = {}
        if self.cross_fitting:
            cross_fitted_models = {
                key: deepcopy(value) for key, value in self.models.items()
            }

        self.cross_fitted_models = cross_fitted_models

        COORDS = {"coeffs": self.labels, "obs_indx": self.index}
        self.fit(X, y, treated, coords=COORDS)

        # Estimate CATE
        self.cate = self.predict_cate(X)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        coords: Dict[str, Any] = None,
    ):
        # Split data to two independent samples of equal size
        (X0, X1, y0, y1, treated0, treated1) = train_test_split(
            X, y, treated, stratify=treated, test_size=0.5
        )

        treated_model = self.models["treated"]
        untreated_model = self.models["untreated"]
        propensity_score_model = self.models["propensity"]
        pseudo_outcome_model = self.models["pseudo_outcome"]

        # Second iteration is the cross-fitting step.
        second_iteration = False
        for _ in range(2):
            # Split data to treated and untreated subsets
            X_t, y_t = X0[treated0 == 1], y0[treated0 == 1]
            X_u, y_u = X0[treated0 == 0], y0[treated0 == 0]

            # Estimate response functions
            _fit(treated_model, X_t, y_t, coords)
            _fit(untreated_model, X_u, y_u, coords)

            # Fit propensity score model
            _fit(propensity_score_model, X, treated, coords)

            g = propensity_score_model.predict_proba(X1)[:, 1]
            mu_0 = untreated_model.predict(X1)
            mu_1 = treated_model.predict(X1)
            mu_w = np.where(treated1 == 0, mu_0, mu_1)

            pseudo_outcome = (treated1 - g) / (g * (1 - g)) * (y1 - mu_w) + mu_1 - mu_0

            # Fit pseudo-outcome model
            _fit(pseudo_outcome_model, X1, pseudo_outcome, coords)

            if self.cross_fitting and not second_iteration:
                # Swap data and estimators
                (X0, X1) = (X1, X0)
                (y0, y1) = (y1, y0)
                (treated0, treated1) = (treated1, treated0)

                treated_model = self.cross_fitted_models["treated"]
                untreated_model = self.cross_fitted_models["untreated"]
                propensity_score_model = self.cross_fitted_models["propensity"]
                pseudo_outcome_model = self.cross_fitted_models["pseudo_outcome"]
                second_iteration = True
            else:
                return self

        return self

    def predict_cate(self, X: pd.DataFrame) -> pd.Series:
        pred = self.models["pseudo_outcome"].predict(X)

        if self.cross_fitting:
            pred2 = self.cross_fitted_models["pseudo_outcome"].predict(X)
            pred = (pred + pred2) / 2

        return pd.Series(pred, index=self.index)

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ) -> Dict[str, np.float64]:
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]

        g = self.models["propensity"].predict_proba(X)[:, 1]
        mu_0 = self.models["untreated"].predict(X)
        mu_1 = self.models["treated"].predict(X)
        mu_w = np.where(treated == 0, mu_0, mu_1)

        pseudo_outcome = (treated - g) / (g * (1 - g)) * (y - mu_w) + mu_1 - mu_0

        return {
            "treated": self.models["treated"].score(X_t, y_t),
            "untreated": self.models["untreated"].score(X_u, y_u),
            "propensity": self.models["propensity"].score(X, treated),
            "pseudo-outcome": self.models["pseudo_outcome"].score(X, pseudo_outcome),
        }
