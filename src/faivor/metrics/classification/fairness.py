from typing import List
from sklearn import metrics as skm
from torchmetrics import Accuracy, F1Score, Precision, Recall
import torch
__all__ = ["ClassificationFairnessMetrics"]


class ClassificationFairnessMetricsMeta(type):
    """Metaclass for dynamically creating classification fairness metric classes."""

    _WHITELISTED_METRICS: List[str] = [
        "accuracy_score", # useful for group fairness comparisons
    ]

    def __new__(mcs, name, bases, dct):
         """Creates a new class, inheriting from skm metrics."""
         for metric_name in mcs._WHITELISTED_METRICS:
            metric_function = getattr(skm, metric_name, None)
            if metric_function:
                def method_wrapper(self, y_true, y_pred, **kwargs):
                    return metric_function(y_true, y_pred, **kwargs)
                dct[metric_name] = method_wrapper
         
         for metric_name in ["accuracy", "f1_score", "precision", "recall"]:
            if metric_name == "accuracy":
                metric_class = Accuracy
            elif metric_name == "f1_score":
                metric_class = F1Score
            elif metric_name == "precision":
                metric_class = Precision
            elif metric_name == "recall":
                metric_class = Recall

            def torchmetrics_method_wrapper(self, y_true, y_pred, **kwargs):
                    metric = metric_class(task = "binary", **kwargs)
                    return metric(
                        torch.tensor(y_pred, dtype = torch.float32),
                        torch.tensor(y_true, dtype= torch.int),
                    ).detach().cpu().item()
            dct[metric_name] = torchmetrics_method_wrapper
         return super().__new__(mcs, name, bases, dct)


class BaseClassificationFairnessMetrics:
    """Base class for classification fairness metrics."""
    pass


class ClassificationFairnessMetrics(BaseClassificationFairnessMetrics, metaclass=ClassificationFairnessMetricsMeta):
    """Class for classification fairness metrics."""

    def custom_disparate_impact(self, y_true, y_pred, sensitive_attribute):
        """Calculates Disparate Impact for classification."""
        import numpy as np
        y_true, y_pred, sensitive_attribute = np.asarray(y_true), np.asarray(y_pred), np.asarray(sensitive_attribute)

        unique_sensitive_values = np.unique(sensitive_attribute)
        if len(unique_sensitive_values) < 2:
            return np.nan

        group_positive_rates = []
        for value in unique_sensitive_values:
            group_mask = sensitive_attribute == value
            if group_mask.sum() == 0:
                group_positive_rates.append(np.nan)
            else:
                 group_positive_rates.append(np.mean(y_pred[group_mask] == np.max(y_pred)))  # Assuming 1 is the positive class

        group_positive_rates = np.asarray(group_positive_rates)
        if np.isnan(group_positive_rates).any():
             return np.nan

        return np.min(group_positive_rates) / np.max(group_positive_rates)

