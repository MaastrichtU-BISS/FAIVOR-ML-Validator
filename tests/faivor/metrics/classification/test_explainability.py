import unittest
import numpy as np
import torch
from src.faivor.metrics.classification.explainability import ClassificationExplainabilityMetrics


class TestClassificationExplainabilityMetrics(unittest.TestCase):

    def setUp(self):
         # Sample classification data
        self.y_true_class = np.array([0, 1, 1, 0, 1, 0])
        self.y_pred_class = np.array([0, 1, 0, 0, 1, 1])
        self.probabilities_class = np.array([
            [0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.7, 0.3],
            [0.2, 0.8]
        ])

        self.metrics = ClassificationExplainabilityMetrics()

    def test_all_explainability_metrics(self):
        for name in dir(self.metrics):
            if not name.startswith("_") and callable(getattr(self.metrics, name)):
                try:
                    if name == "custom_prediction_entropy":
                        result = getattr(self.metrics, name)(self.probabilities_class)
                    else:
                        result = getattr(self.metrics, name)(self.y_true_class, self.y_pred_class)
                    self.assertIsNotNone(result, f"Metric {name} returned None")
                except Exception as e:
                    self.fail(f"Metric {name} raised an exception: {e}")



if __name__ == '__main__':
    unittest.main()