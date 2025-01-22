import pytest
import unittest
import numpy as np
import torch
from faivor.metrics.classification.fairness import ClassificationFairnessMetrics

class TestClassificationFairnessMetrics(unittest.TestCase):

    def setUp(self):
        # Sample Classification Data (Same as in original test.py, but smaller and more suitable for unit tests)
        self.y_true_class = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        self.y_pred_class = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        self.sensitive_attribute_class = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        self.metrics = ClassificationFairnessMetrics()

    def test_all_fairness_metrics(self):
        for name in dir(self.metrics):
            if not name.startswith("_") and callable(getattr(self.metrics, name)):
                try:
                     if name == "custom_disparate_impact":
                        result = getattr(self.metrics, name)(self.y_true_class, self.y_pred_class, self.sensitive_attribute_class)
                     else:
                         result = getattr(self.metrics, name)(self.y_true_class, self.y_pred_class)
                     assert result is not None, f"Metric {name} returned None"
                except Exception as e:
                      pytest.fail(f"Metric {name} raised an exception: {e}")