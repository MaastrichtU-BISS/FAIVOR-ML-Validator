import inspect
from typing import List
from sklearn import metrics as skm


__all__ = ["ClassificationPerformanceMetrics"]


class ClassificationPerformanceMetricsMeta(type):
    """Metaclass for dynamically creating classification performance metric classes."""

    _WHITELISTED_METRICS: List[str] = [
        "accuracy_score",
        "balanced_accuracy_score",
        "average_precision_score",
        "f1_score",
        "precision_score",
        "recall_score",
        "roc_auc_score",
        "jaccard_score",
        "log_loss",
        "matthews_corrcoef",
        "brier_score_loss",
        "top_k_accuracy_score",
        "roc_curve",
        "precision_recall_curve",
        "hamming_loss",
        "zero_one_loss",
        "confusion_matrix"
    ]

    def __new__(mcs, name, bases, dct):
        """Creates a new class, inheriting from sklearn.metrics."""
        
        def create_metric_function(metric_name):
            """Factory function to create a metric wrapper."""
            metric_function = getattr(skm, metric_name, None)
            if metric_function is None:
                raise ValueError(f"Metric '{metric_name}' not found in sklearn.metrics.")

            def method_wrapper(self, y_true, y_pred, **kwargs):
                """Wrapper for sklearn.metrics functions."""
                return metric_function(y_true, y_pred, **kwargs)

            method_wrapper.__name__ = metric_name
            method_wrapper.__doc__ = metric_function.__doc__
            return method_wrapper

        # Dynamically add methods for whitelisted metrics
        for metric_name in mcs._WHITELISTED_METRICS:
            if hasattr(skm, metric_name):  # Ensure the metric exists in sklearn.metrics
                dct[metric_name] = create_metric_function(metric_name)

        return super().__new__(mcs, name, bases, dct)


class BaseClassificationPerformanceMetrics:
    """Base class for classification performance metrics."""
    pass


class ClassificationPerformanceMetrics(BaseClassificationPerformanceMetrics, metaclass=ClassificationPerformanceMetricsMeta):
    """Class for classification performance metrics."""

    def custom_error_rate(self, y_true, y_pred):
        """Calculates custom error rate for classification."""
        import numpy as np
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return 1 - skm.accuracy_score(y_true, y_pred)

