import pytest
import numpy as np
import torch
from sklearn import metrics as skm
from faivor.metrics.classification.performance import ClassificationPerformanceMetrics


# Sample Classification Data
y_true_class = np.array([0, 1, 1, 0, 1, 0])
y_pred_class = np.array([0, 1, 0, 0, 1, 1])

metrics = ClassificationPerformanceMetrics()

def test_all_performance_metrics():
    for name in dir(metrics):
        if not name.startswith("_") and callable(getattr(metrics, name)):
            try:
                if name == "custom_error_rate":
                    result = getattr(metrics, name)(y_true_class, y_pred_class)
                elif name == "accuracy_score":
                    result = getattr(metrics, name)(y_true_class, y_pred_class)
                    assert result  == skm.accuracy_score(y_true_class, y_pred_class)
                else:
                    result = getattr(metrics, name)(y_true_class, y_pred_class)
                assert result is not None, f"Metric {name} returned None"

            except Exception as e:
                    pytest.fail(f"Metric {name} raised an exception: {e}")