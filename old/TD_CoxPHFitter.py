from lifelines import CoxPHFitter
import pandas as pd


class TD_CoxPHFitter:
    def __init__(self, penalizer: float | int = 0.01, l1_ratio: float = 0.0):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self._cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    
    def fit(self, X_train: pd.DataFrame, duration_col: str, event_col: str, **kwargs):
        self._cph.fit(X_train, duration_col, event_col=event_col, **kwargs)

    def score(self, X_test: pd.DataFrame, scoring_method: str = "log_likelihood", **kwargs):
        return self._cph.score(X_test, scoring_method, **kwargs)
    
    def plot(self, ax):
        self._cph.plot(ax=ax)

    @property
    def concordance_index(self):
        if not hasattr(self._cph, "concordance_index_"):
            raise ValueError("CoxPHFitter not fitted yet. Fit it first")
        return self._cph.concordance_index_
    
    @property
    def cph(self):
        return self._cph