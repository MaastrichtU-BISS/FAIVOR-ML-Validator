from typing import List
from sklearn import metrics as skm
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
import torch
__all__ = ["RegressionFairnessMetrics"]


class RegressionFairnessMetricsMeta(type):
    """Metaclass for dynamically creating regression fairness metric classes."""

    _WHITELISTED_METRICS: List[str] = [] # No standard fairness metrics in sklearn for regression, may need custom implementation

    def __new__(mcs, name, bases, dct):
         """Creates a new class, inheriting from skm metrics."""
         for metric_name in mcs._WHITELISTED_METRICS:
            metric_function = getattr(skm, metric_name, None)
            if metric_function:
                def method_wrapper(self, y_true, y_pred, **kwargs):
                    return metric_function(y_true, y_pred, **kwargs)
                dct[metric_name] = method_wrapper
         
         for metric_name in ["mean_absolute_error", "mean_squared_error", "mean_absolute_percentage_error"]:
            if metric_name == "mean_absolute_error":
                metric_class = MeanAbsoluteError
            elif metric_name == "mean_squared_error":
                metric_class = MeanSquaredError
            elif metric_name == "mean_absolute_percentage_error":
               metric_class = MeanAbsolutePercentageError

            def torchmetrics_method_wrapper(self, y_true, y_pred, **kwargs):
                    metric = metric_class(**kwargs)
                    return metric(
                        torch.tensor(y_pred, dtype = torch.float32),
                        torch.tensor(y_true, dtype= torch.float32),
                    ).detach().cpu().item()
            dct[metric_name] = torchmetrics_method_wrapper
         return super().__new__(mcs, name, bases, dct)


class BaseRegressionFairnessMetrics:
    """Base class for regression fairness metrics."""
    pass


class RegressionFairnessMetrics(BaseRegressionFairnessMetrics, metaclass=RegressionFairnessMetricsMeta):
    """Class for regression fairness metrics."""

    def custom_demographic_parity_ratio(self, y_true, y_pred, sensitive_attribute):
        """
            Calculates Demographic Parity Ratio for regression
        """
        import numpy as np
        y_true, y_pred, sensitive_attribute = np.asarray(y_true), np.asarray(y_pred), np.asarray(sensitive_attribute)

        unique_sensitive_values = np.unique(sensitive_attribute)
        if len(unique_sensitive_values) < 2:
            return np.nan # not applicable for less than 2 groups

        group_means = []
        for value in unique_sensitive_values:
            group_mask = sensitive_attribute == value
            if group_mask.sum() == 0:
                group_means.append(np.nan) # to handle potential nan group mean
            else:
                group_means.append(np.mean(y_pred[group_mask]))

        group_means = np.asarray(group_means)
        if np.isnan(group_means).any():
             return np.nan # to handle nan group means

        return np.min(group_means) / np.max(group_means)
