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
        """Creates a new class, inheriting from skm metrics."""
        for metric_name in mcs._WHITELISTED_METRICS:
            metric_function = getattr(skm, metric_name, None)
            if metric_function:
                def method_wrapper(self, y_true, y_pred, **kwargs):
                    return metric_function(y_true, y_pred, **kwargs)
                dct[metric_name] = method_wrapper
        return super().__new__(mcs, name, bases, dct)


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

