import pytest
import numpy as np
import torch
from faivor.metrics.regression.performance import RegressionPerformanceMetrics


def setUp(self):
        # Sample Regression Data (Same as in original test.py, but smaller and more suitable for unit tests)
    self.y_true_reg = np.array([3, -0.5, 2, 7, 4.2, 1])
    self.y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8, 3.9, 1.1])

    self.metrics = RegressionPerformanceMetrics()

def test_all_performance_metrics(self):
    for name in dir(self.metrics):
        if not name.startswith("_") and callable(getattr(self.metrics, name)):
            try:
                if name == "custom_mean_percentage_error":
                        result = getattr(self.metrics, name)(self.y_true_reg, self.y_pred_reg)
                else:
                        result = getattr(self.metrics, name)(self.y_true_reg, self.y_pred_reg)
                assert result is not None, f"Metric {name} returned None"
            except Exception as e:
                    pytest.fail(f"Metric {name} raised an exception: {e}")