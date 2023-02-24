import pandas as pd
import numpy as np
from sklearn.utils import check_consistent_length
from sklearn.base import clone
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def check_shapes(X, y, treated):
    check_consistent_length(X, y, treated)
    if len(y.shape) > 1:
        raise(ValueError("Target variable shape"))


class MetaLearner:
    def __init__(
        self,
        X, y,
        treated
    ):
        check_consistent_length(X, y, treated)
        
        self.treated = treated
        self.X = X
        self.y = y
        
    def predict_cate(self, X):
        raise(NotImplementedError())
        
    def predict_ate(self, X):
        return self.predict_cate(X).mean()
    
    def summary(self):
        raise(NotImplementedError())
    
    def plot(self):
        raise(NotImplementedError())
        

class SLearner(MetaLearner):
    def __init__(self, 
                 X,
                 y,
                 treated,
                 model):
        super().__init__(X=X, y=y, treated=treated)
        X_T = pd.concat([X, treated], axis=1)
        
        self.model = model.fit(X_T, y)
        self.cate = self.predict_cate(X)
    
    def predict_cate(self, X):
        X_control = pd.concat([X, 0 * treated], axis=1)
        X_treated = pd.concat([X, 0 * treated + 1], axis=1)
        return self.model.predict(X_treated) - self.model.predict(X_control)


class TLearner(MetaLearner):
    def __init__(self, 
                 X,
                 y,
                 treated,
                 model=None,
                 treated_model=None,
                 untreated_model=None
                ):
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
        
        self.untreated_model = untreated_model.fit(X[treated==0], y[treated==0])
        self.treated_model = treated_model.fit(X[treated==1], y[treated==1])
        self.cate = self.predict_cate(X)
    
    def predict_cate(self, X):
        return self.treated_model.predict(X) - self.untreated_model.predict(X)

    
class XLearner(MetaLearner):
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
            propensity_score_model = LogisticRegression(penalty='none')
        
        if model is not None:
            treated_model = clone(model)
            untreated_model = clone(model)
            treated_cate_estimator = clone(model)
            untreated_cate_estimator = clone(model)
 
        # Split data to treated and untreated subsets
        X_t, y_t = X[treated==1], y[treated==1]
        X_u, y_u = X[treated==0], y[treated==0]
        
        # Estimate response function
        self.models = {'treated': treated_model.fit(X_t, y_t),
                       'untreated': untreated_model.fit(X_u, y_u)}
        
        tau_t = y_t - untreated_model.predict(X_t)
        tau_u = treated_model.predict(X_u) - y_u
        
        # Estimate CATE separately on treated and untreated subsets
        self._imputed_treatment_effects = {'treated': tau_t,
                                           'untreated': tau_u}
        
        self.models['treated_cate'] = treated_cate_estimator.fit(X_t, tau_t)
        self.models['untreated_cate'] = untreated_cate_estimator.fit(X_u, tau_u)
        
        cate_estimate_treated = treated_cate_estimator.predict(X)
        cate_estimate_untreated = treated_cate_estimator.predict(X)
        
        self.cate_estimates = {'treated': cate_estimate_treated,
                               'untreated': cate_estimate_untreated}
        
        # Find a weight function g
        self.models['propensity_score'] = propensity_score_model.fit(X, treated)
        g = propensity_score_model.predict(X)
        
        # Average CATE estimates using g
        self.cate = g * cate_estimate_untreated + (1 - g) * cate_estimate_treated
    
    
    def predict_cate(self, X):
        cate_estimate_treated = self.models['treated_cate'].predict(X)
        cate_estimate_untreated = self.models['untreated_cate'].predict(X)
        
        g = self.models['propensity_score'].predict(X)
        
        return g * cate_estimate_untreated + (1 - g) * cate_estimate_treated