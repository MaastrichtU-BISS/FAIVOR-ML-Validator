from typing import List
from sklearn import metrics as skm

__all__ = ["RegressionExplainabilityMetrics"]

class RegressionExplainabilityMetricsMeta(type):
    """Metaclass for dynamically creating regression explainability metric classes."""
    _WHITELISTED_METRICS: List[str] = [] # No standard explainability metrics in sklearn directly.

    def __new__(mcs, name, bases, dct):
        """Creates a new class, inheriting from skm metrics."""
        for metric_name in mcs._WHITELISTED_METRICS:
            metric_function = getattr(skm, metric_name, None)
            if metric_function:
                def method_wrapper(self, y_true, y_pred, **kwargs):
                    return metric_function(y_true, y_pred, **kwargs)
                dct[metric_name] = method_wrapper
        return super().__new__(mcs, name, bases, dct)


class BaseRegressionExplainabilityMetrics:
    """Base class for regression explainability metrics."""
    pass


class RegressionExplainabilityMetrics(BaseRegressionExplainabilityMetrics, metaclass=RegressionExplainabilityMetricsMeta):
    """Class for regression explainability metrics."""

    def custom_feature_importance_ratio(self, feature_importances):
        """
        Calculate a ratio to assess feature importance
        """
        import numpy as np
        feature_importances = np.asarray(feature_importances)
        if len(feature_importances) == 0:
            return np.nan
        return np.min(feature_importances)/np.max(feature_importances)
