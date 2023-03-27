"PyMC based meta-learners."
from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xarray.core.dataarray import DataArray

from causalpy.pymc_models import LogisticRegression, ModelBuilder
from causalpy.skl_meta_learners import (
    DRLearner,
    MetaLearner,
    SLearner,
    TLearner,
    XLearner,
)
from causalpy.utils import _fit


class BayesianMetaLearner(MetaLearner):
    "Base class for PyMC based meta-learners."

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        coords: Dict[str, Any] = None,
    ):
        """
        Fits base-learners.

        Parameters
        ----------
        X :     pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y :     pandas.Series of shape (n_samples, ).
                Target vector.
        treated :   pandas.Series of shape (n_samples, ).
        coords : Dict[str, Any].
            Dictionary containing the keys coeffs and obs_indx.
        """
        raise NotImplementedError()

    def predict_cate(self, X: pd.DataFrame) -> DataArray:
        """
        Predicts distribution of treatement effect.

        Parameters
        ----------
        X :     pandas.DataFrame of shape (n_samples, n_featues).
                Feature matrix.
        """
        raise NotImplementedError()


class BayesianSLearner(SLearner, BayesianMetaLearner):
    """
    Implements PyMC version of S-learner described in [1]. S-learner estimates
    conditional average treatment effect with the use of a single model.

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
    model : causalpy.pymc_models.ModelBuilder.
            Base learner.
    """

    def predict_cate(self, X: pd.DataFrame) -> DataArray:
        X_untreated = X.assign(treatment=0)
        X_treated = X.assign(treatment=1)
        m = self.models["model"]

        pred_treated = m.predict(X_treated)["posterior_predictive"].mu
        pred_untreated = m.predict(X_untreated)["posterior_predictive"].mu

        return pred_treated - pred_untreated


class BayesianTLearner(TLearner, BayesianMetaLearner):
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
    model : causalpy.pymc_models.ModelBuilder.
            If specified, it will be used both as treated and untreated model. Either
            model or both of treated_model and untreated_model have to be specified.
    treated_model: causalpy.pymc_models.ModelBuilder.
            Model used for predicting target vector for treated values.
    untreated_model: causalpy.pymc_models.ModelBuilder.
            Model used for predicting target vector for untreated values.
    """

    def predict_cate(self, X: pd.DataFrame) -> DataArray:
        treated_model = self.models["treated"]
        untreated_model = self.models["untreated"]

        pred_treated = treated_model.predict(X)["posterior_predictive"].mu
        pred_untreated = untreated_model.predict(X)["posterior_predictive"].mu

        return pred_treated - pred_untreated


class BayesianXLearner(XLearner, BayesianMetaLearner):
    """
    Implements of PyMC version of X-learner introduced in [1]. X-learner estimates
    conditional average treatment effect with the use of five separate models.

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
    model : causalpy.pymc_models.ModelBuilder.
            If specified, it will be used in all of the subregressions, except for the
            propensity_score_model. Either model or all of treated_model,
            untreated_model, treated_cate_estimator and untreated_cate_estimator have
            to be specified.
    treated_model : causalpy.pymc_models.ModelBuilder.
            Model used for predicting target vector for treated values.
    untreated_model :   causalpy.pymc_models.ModelBuilder.
            Model used for predicting target vector for untreated values.
    untreated_cate_estimator :  causalpy.pymc_models.ModelBuilder.
            Model used for CATE estimation on untreated data.
    treated_cate_estimator :    causalpy.pymc_models.ModelBuilder.
            Model used for CATE estimation on treated data.
    propensity_score_model :    causalpy.pymc_models.ModelBuilder,
                                default = causalpy.pymc_models.LogisticRegression().
            Model used for propensity score estimation. Output values should be in the
            interval [0, 1].
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        model: Optional[ModelBuilder] = None,
        treated_model: Optional[ModelBuilder] = None,
        untreated_model: Optional[ModelBuilder] = None,
        treated_cate_estimator: Optional[ModelBuilder] = None,
        untreated_cate_estimator: Optional[ModelBuilder] = None,
        propensity_score_model: Optional[ModelBuilder] = None,
    ):
        super().__init__(
            X,
            y,
            treated,
            model,
            treated_model,
            untreated_model,
            treated_cate_estimator,
            untreated_cate_estimator,
            propensity_score_model,
        )

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

        pred_u_t = az.extract(
            untreated_model.predict(X_t), group="posterior_predictive", var_names="mu"
        ).mean(axis=1)
        pred_t_u = az.extract(
            treated_model.predict(X_u), group="posterior_predictive", var_names="mu"
        ).mean(axis=1)

        tau_t = y_t - pred_u_t
        tau_u = - y_u + pred_t_u

        # Estimate CATE separately on treated and untreated subsets
        _fit(treated_cate_estimator, X_t, tau_t, coords)
        _fit(untreated_cate_estimator, X_u, tau_u, coords)

        # Fit propensity score model
        _fit(propensity_score_model, X, treated, coords)
        return self

    def predict_cate(self, X: pd.DataFrame) -> DataArray:
        treated_model = self.models["treated_cate"]
        untreated_model = self.models["untreated_cate"]

        cate_estimate_treated = treated_model.predict(X)["posterior_predictive"].mu
        cate_estimate_untreated = untreated_model.predict(X)["posterior_predictive"].mu
        g = self.models["propensity"].predict(X)["posterior_predictive"].mu

        return (1 - g) * cate_estimate_untreated + g * cate_estimate_treated


class BayesianDRLearner(DRLearner, BayesianMetaLearner):
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
        model : causalpy.pymc_models.ModelBuilder.
                If specified, it will be used in all of the subregressions, except for
                the propensity_score_model. Either model or all of treated_model,
                untreated_model, treated_cate_estimator and untreated_cate_estimator
                have to be specified.
        treated_model : causalpy.pymc_models.ModelBuilder.
                Model used for predicting target vector for treated values.
        untreated_model :   causalpy.pymc_models.ModelBuilder.
                Model used for predicting target vector for untreated values.
        pseudo_outcome_model :  causalpy.pymc_models.ModelBuilder.
                Model used for pseudo-outcome estimation.
        propensity_score_model :    causalpy.pymc_models.ModelBuilder,
                                default = causalpy.pymc_models.LogisticRegression().
                Model used for propensity score estimation.
        cross_fitting : bool, default=False.
                If True, performs a cross fitting step.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        model: Optional[ModelBuilder] = None,
        treated_model: Optional[ModelBuilder] = None,
        untreated_model: Optional[ModelBuilder] = None,
        pseudo_outcome_model: Optional[ModelBuilder] = None,
        propensity_score_model: Optional[ModelBuilder] = LogisticRegression(),
        cross_fitting: bool = False,
    ):
        super().__init__(
            X,
            y,
            treated,
            model,
            treated_model,
            untreated_model,
            pseudo_outcome_model,
            propensity_score_model,
            cross_fitting,
        )

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
            _fit(propensity_score_model, X0, treated0, coords)

            g = az.extract(
                propensity_score_model.predict(X1),
                group="posterior_predictive",
                var_names="mu",
            ).mean(axis=1)
            mu_0 = az.extract(
                untreated_model.predict(X1),
                group="posterior_predictive",
                var_names="mu",
            ).mean(axis=1)
            mu_1 = az.extract(
                treated_model.predict(X1), group="posterior_predictive", var_names="mu"
            ).mean(axis=1)
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

    def predict_cate(self, X: pd.DataFrame) -> DataArray:
        pred = self.models["pseudo_outcome"].predict(X)["posterior_predictive"].mu

        if self.cross_fitting:
            pred2 = (
                self.cross_fitted_models["pseudo_outcome"]
                .predict(X)["posterior_predictive"]
                .mu
            )
            pred = (pred + pred2) / 2

        return pred
