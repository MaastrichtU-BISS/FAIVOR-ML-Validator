import pytest
import numpy as np
import torch
from faivor.metrics.regression.performance import RegressionPerformanceMetrics


# Sample Regression Data
y_true_reg = np.array([3, -0.5, 2, 7, 4.2, 1])
y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8, 3.9, 1.1])

metrics = RegressionPerformanceMetrics()

def test_all_performance_metrics():
    for name in dir(metrics):
        if not name.startswith("_") and callable(getattr(metrics, name)):
            try:
                if name == "custom_mean_percentage_error":
                        result = getattr(metrics, name)(y_true_reg, y_pred_reg)
                else:
                        result = getattr(metrics, name)(y_true_reg, y_pred_reg)
                assert result is not None, f"Metric {name} returned None"
            except Exception as e:
                    pytest.fail(f"Metric {name} raised an exception: {e}")