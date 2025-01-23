import pytest
import numpy as np
import torch
from faivor.metrics.classification.performance import ClassificationPerformanceMetrics


def setUp(self):
    # Sample Classification Data (Same as in original test.py, but smaller and more suitable for unit tests)
    self.y_true_class = np.array([0, 1, 1, 0, 1, 0])
    self.y_pred_class = np.array([0, 1, 0, 0, 1, 1])

    self.metrics = ClassificationPerformanceMetrics()

def test_all_performance_metrics(self):
    for name in dir(self.metrics):
        if not name.startswith("_") and callable(getattr(self.metrics, name)):
            try:
                if name == "custom_error_rate":
                    result = getattr(self.metrics, name)(self.y_true_class, self.y_pred_class)
                else:
                    result = getattr(self.metrics, name)(self.y_true_class, self.y_pred_class)
                assert result is not None, f"Metric {name} returned None"

            except Exception as e:
                    pytest.fail(f"Metric {name} raised an exception: {e}")