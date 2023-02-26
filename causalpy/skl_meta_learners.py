import pandas as pd
import numpy as np
from sklearn.utils import check_consistent_length
from sklearn.base import clone
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.base import RegressorMixin
from utils import _is_variable_dummy_coded


class MetaLearner:
    """
    Base class for meta-learners. 
    """
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series
    ) -> None:
        # Check whether input is appropriate
        check_consistent_length(X, y, treated)
        if not _is_variable_dummy_coded(treated):
            raise(ValueError('Treatment variable is not dummy coded.'))

        self.treated = treated
        self.X = X
        self.y = y
        

    def predict_cate(self, X: pd.DataFrame) -> np.array:
        """
        Predict conditional average treatment effect for given input X.
        """
        raise(NotImplementedError())
        

    def predict_ate(self, X: pd.DataFrame) -> np.float64:
        """
        Predict average treatment effect for given input X.
        """
        return self.predict_cate(X).mean()
    
    def bootstrap(self,
                  X_ins: pd.DataFrame,
                  y: pd.Series,
                  treated: pd.Series,
                  X: None,
                  frac_samples: float = None,
                  n_samples: int = 1000,
                  n_iter: int = 1000
                 ) -> np.array:
        """
        Runs bootstrap n_iter times on a sample of size n_samples. Fits on (X_ins, y, treated),
        then predicts on X.
        """
        results = []
        for i in range(n_iter):
            X_bs = X_ins.sample(frac=frac_samples,
                                n=n_samples,
                                replace=True)
            y_bs = y.loc[X_bs.index].reset_index(drop=True)
            t_bs = treated.loc[X_bs.index].reset_index(drop=True)

            self.fit(X_bs.reset_index(drop=True), y_bs, t_bs)  # This overwrites self.models!
            results.append(self.predict_cate(X))
        
        return np.array(results)

    def ate_confidence_interval(self,
                                X_ins: pd.DataFrame,
                                y: pd.Series,
                                treated: pd.Series,
                                X: None,
                                q: float = .95,
                                frac_samples: float = None,
                                n_samples: int = 1000,
                                n_iter: int = 1000):
        """
        Estimates confidence intervals for ATE on X using bootstraping. 
        """
        cates = self.bootstrap(X_ins,
                               y,
                               treated,
                               X,
                               frac_samples,
                               n_samples,
                               n_iter)
        return np.quantile(cates, q=q), np.quantile(cates, q=1-q)

    def cate_confidence_interval(self,
                                 X_ins: pd.DataFrame,
                                 y: pd.Series,
                                 treated: pd.Series,
                                 X: None,
                                 q: float = .95,
                                 frac_samples: float = None,
                                 n_samples: int = 1000,
                                 n_iter: int = 1000):
        """
        Estimates confidence intervals for CATE on X using bootstraping. 
        """
        cates = self.bootstrap(X_ins,
                               y,
                               treated,
                               X,
                               frac_samples,
                               n_samples,
                               n_iter)
        conf_ints = np.append(np.quantile(cates, .9, axis=0).reshape(-1,1),
                              np.quantile(cates, .1, axis=0).reshape(-1, 1),
                              axis=1)
        return conf_ints


    def fit(self, X: pd.DataFrame,
            y: pd.Series,
            treated: pd.Series):
        "Fits model."
        raise(NotImplementedError())


    def summary(self):
        raise(NotImplementedError())
    

    def plot(self):
        raise(NotImplementedError())
        

class SLearner(MetaLearner):
    """
    Implements of S-learner described in [1]. S-learner estimates conditional average treatment effect 
    with the use of a single model. 
    
    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. 
        Metalearners for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    """
    def __init__(self, 
                 X: pd.DataFrame,
                 y: pd.Series,
                 treated: pd.Series,
                 model) -> None:
        super().__init__(X=X, y=y, treated=treated)
        self.model = model
        self.fit(X, y, treated)
        self.cate = self.predict_cate(X)
    
    def fit(self, X: pd.DataFrame,
            y: pd.Series,
            treated: pd.Series):
        X_T = X.assign(treatment=treated)
        self.model = self.model.fit(X_T, y)
        return self
    
    def predict_cate(self, X: pd.DataFrame) -> np.array:
        X_control = X.assign(treatment=0)
        X_treated = X.assign(treatment=1)
        return self.model.predict(X_treated) - self.model.predict(X_control)


class TLearner(MetaLearner):
    """
    Implements of T-learner described in [1]. T-learner fits two separate models to estimate
    conditional average treatment effect. 
    
    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. 
        Metalearners for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    """
    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 treated: pd.Series,
                 model=None,
                 treated_model=None,
                 untreated_model=None
                ) -> None:
        super().__init__(X=X, y=y, treated=treated)
        
        if model is None and (untreated_model is None or treated_model is None):
            raise(ValueError("Either model or both of treated_model and untreated_model \
            have to be specified."))
        elif not (model is None or untreated_model is None or treated_model is None):
            raise(ValueError("Either model or both of treated_model and untreated_model \
            have to be specified."))
        
        if model is not None:
            untreated_model = clone(model)
            treated_model = clone(model)
        
        self.models = {'treated': treated_model,
                       'untreated': untreated_model}
        
        self.fit(X, y, treated)
        self.cate = self.predict_cate(X)
    
    def fit(self, X: pd.DataFrame,
            y: pd.Series,
            treated: pd.Series):
        self.models['treated'].fit(X[treated==1], y[treated==1])
        self.models['untreated'].fit(X[treated==0], y[treated==0])
        return self


    def predict_cate(self, X: pd.DataFrame) -> np.array:
        treated_model = self.models['treated']
        untreated_model = self.models['untreated']
        return treated_model.predict(X) - untreated_model.predict(X)

    
class XLearner(MetaLearner):
    """
    Implements of X-learner introduced in [1]. X-learner estimates conditional average treatment
    effect with the use of five separate models. 
    
    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. 
        Metalearners for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    """
    def __init__(self, 
                 X,
                 y,
                 treated,
                 model=None,
                 treated_model=None,
                 untreated_model=None,
                 treated_cate_estimator=None,
                 untreated_cate_estimator=None,
                 propensity_score_model=None
                ):
        super().__init__(X=X, y=y, treated=treated)
        
        if model is None and (untreated_model is None or treated_model is None):
            raise(ValueError("""Either model or each of treated_model, untreated_model, \
            treated_cate_estimator, untreated_cate_estimator has to be specified."""))
        elif not (model is None or untreated_model is None or treated_model is None):
            raise(ValueError("Either model or each of treated_model, untreated_model, \
            treated_cate_estimator, untreated_cate_estimator has to be specified."))
        
        if propensity_score_model is None:
            propensity_score_model = LogisticRegression(penalty=None)
        
        if model is not None:
            treated_model = clone(model)
            untreated_model = clone(model)
            treated_cate_estimator = clone(model)
            untreated_cate_estimator = clone(model)
        
        self.models = {'treated': treated_model,
                       'untreated': untreated_model,
                       'treated_cate': treated_cate_estimator,
                       'untreated_cate': untreated_cate_estimator,
                       'propensity': propensity_score_model
                      }

        self.fit(X, y, treated)
        
        # Compute cate
        cate_t = treated_cate_estimator.predict(X)
        cate_u = treated_cate_estimator.predict(X)
        g = self.models['propensity'].predict(X)
        
        self.cate = g * cate_u + (1 - g) * cate_t
    
    def fit(self, X: pd.DataFrame,
            y: pd.Series,
            treated: pd.Series):
        (treated_model,
         untreated_model,
         treated_cate_estimator,
         untreated_cate_estimator,
         propensity_score_model) = self.models.values()

        # Split data to treated and untreated subsets
        X_t, y_t = X[treated==1], y[treated==1]
        X_u, y_u = X[treated==0], y[treated==0]
        
        # Estimate response function
        treated_model.fit(X_t, y_t)
        untreated_model.fit(X_u, y_u)
        
        tau_t = y_t - untreated_model.predict(X_t)
        tau_u = treated_model.predict(X_u) - y_u
        
        # Estimate CATE separately on treated and untreated subsets
        treated_cate_estimator.fit(X_t, tau_t)
        untreated_cate_estimator.fit(X_u, tau_u)
        
        # Fit propensity score model
        propensity_score_model.fit(X, treated)
        return self


    def predict_cate(self, X):
        cate_estimate_treated = self.models['treated_cate'].predict(X)
        cate_estimate_untreated = self.models['untreated_cate'].predict(X)
        
        g = self.models['propensity'].predict_proba(X)[:, 1]
        
        return g * cate_estimate_untreated + (1 - g) * cate_estimate_treated


class DRLearner(MetaLearner):
    """
    Implements of DR-learner also known as doubly robust learner. DR-learner estimates
    conditional average treatment effect with the use of three separate models.

    """
    def __init__(self, 
                 X,
                 y,
                 treated,
                 model=None,
                 treated_model=None,
                 untreated_model=None,
                 propensity_score_model=None
                ):
        super().__init__(X=X, y=y, treated=treated)
        
        if model is None and (untreated_model is None or treated_model is None):
            raise(ValueError("""Either model or each of treated_model, untreated_model, \
            treated_cate_estimator, untreated_cate_estimator has to be specified."""))
        elif not (model is None or untreated_model is None or treated_model is None):
            raise(ValueError("Either model or each of treated_model, untreated_model, \
            treated_cate_estimator, untreated_cate_estimator has to be specified."))
            
        if propensity_score_model is None:
            propensity_score_model = LogisticRegression(penalty=None)
        
        if model is not None:
            treated_model = clone(model)
            untreated_model = clone(model)
        
        # Estimate response function
        self.models = {'treated': treated_model,
                       'untreated': untreated_model,
                       'propensity': propensity_score_model}
        
        self.fit(X, y, treated)
        
        # Estimate CATE
        g = self.models['propensity'].predict_proba(X)[:, 1]
        m0 = untreated_model.predict(X)
        m1 = treated_model.predict(X)

        self.cate = (treated * (y - m1) / g + m1
                    - ((1 - treated) * (y - m0) / (1 - g) + m0))
    

    def fit(self, X: pd.DataFrame,
            y: pd.Series,
            treated: pd.Series):
        # Split data to treated and untreated subsets
        X_t, y_t = X[treated==1], y[treated==1]
        X_u, y_u = X[treated==0], y[treated==0]

        (treated_model,
         untreated_model,
         propensity_score_model) = self.models.values()

        # Estimate response functions 
        treated_model.fit(X_t, y_t)
        untreated_model.fit(X_u, y_u)

        # Fit propensity score model
        propensity_score_model.fit(X, treated)
        
        return self
    
    def predict_cate(self, X):
        return self.cate
