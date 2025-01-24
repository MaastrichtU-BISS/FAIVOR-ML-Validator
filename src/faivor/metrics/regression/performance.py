import inspect
from typing import List
from sklearn import metrics as skm

__all__ = ["RegressionPerformanceMetrics"]


class RegressionPerformanceMetricsMeta(type):
    """Metaclass for dynamically creating regression performance metric classes."""

    _WHITELISTED_METRICS: List[str] = [
        "mean_absolute_error",
        "mean_squared_error",
        "mean_squared_log_error",
        "median_absolute_error",
        "r2_score",
        "explained_variance_score",
        "max_error",
        "mean_poisson_deviance",
        "mean_gamma_deviance",
        "d2_absolute_error_score",
        "mean_pinball_loss"
    ]

    def __new__(mcs, name, bases, dct):
        """Creates a new class, inheriting from sklearn.metrics."""
        for metric_name in mcs._WHITELISTED_METRICS:
            if hasattr(skm, metric_name):  # Ensure the metric exists
                dct[metric_name] = create_metric_wrapper(metric_name)
        return super().__new__(mcs, name, bases, dct)

def create_metric_wrapper(metric_name):
    """Factory function to create a metric wrapper for the given metric name."""
    metric_function = getattr(skm, metric_name, None)
    if metric_function is None:
        raise ValueError(f"Metric '{metric_name}' not found in sklearn.metrics.")

    def method_wrapper(self, y_true, y_pred, **kwargs):
        """Wrapper function for the metric."""
        return metric_function(y_true, y_pred, **kwargs)
    
    method_wrapper.__name__ = metric_name  # Set the method name for clarity
    method_wrapper.__doc__ = metric_function.__doc__  # Use the original docstring
    return method_wrapper

class BaseRegressionPerformanceMetrics:
    """Base class for regression performance metrics."""
    pass


class RegressionPerformanceMetrics(BaseRegressionPerformanceMetrics, metaclass=RegressionPerformanceMetricsMeta):
    """Class for regression performance metrics."""

    def custom_mean_percentage_error(self, y_true, y_pred):
        """Calculates Mean Percentage Error for regression."""
        import numpy as np
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() == 0:
             return np.nan # to avoid division by 0
        return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

