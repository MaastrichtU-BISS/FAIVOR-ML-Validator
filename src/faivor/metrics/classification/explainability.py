from typing import List
from sklearn import metrics as skm

__all__ = ["ClassificationExplainabilityMetrics"]


class ClassificationExplainabilityMetricsMeta(type):
    """Metaclass for dynamically creating classification explainability metric classes."""

    _WHITELISTED_METRICS: List[str] = [] # sklearn doesn't provide direct explainability metrics

    def __new__(mcs, name, bases, dct):
        """Creates a new class, inheriting from skm metrics."""
        for metric_name in mcs._WHITELISTED_METRICS:
            metric_function = getattr(skm, metric_name, None)
            if metric_function:
                def method_wrapper(self, y_true, y_pred, **kwargs):
                    return metric_function(y_true, y_pred, **kwargs)
                dct[metric_name] = method_wrapper
        return super().__new__(mcs, name, bases, dct)


class BaseClassificationExplainabilityMetrics:
    """Base class for classification explainability metrics."""
    pass


class ClassificationExplainabilityMetrics(BaseClassificationExplainabilityMetrics, metaclass=ClassificationExplainabilityMetricsMeta):
    """Class for classification explainability metrics."""

    def custom_prediction_entropy(self, probas):
        """Calculate the average entropy of prediction probabilities."""
        import numpy as np
        probas = np.asarray(probas)
        log_probs = np.log2(probas)
        return -np.mean(np.sum(probas * log_probs, axis=1))
