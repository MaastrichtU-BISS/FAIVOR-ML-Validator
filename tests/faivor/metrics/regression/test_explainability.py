import unittest
import numpy as np
import torch
from src.faivor.metrics.regression.explainability import RegressionExplainabilityMetrics

class TestRegressionExplainabilityMetrics(unittest.TestCase):
    def setUp(self):
        # Sample regression data
         self.feature_importances_reg = np.array([0.1, 0.2, 0.7, 0.05, 0.05])
         self.y_true_reg = np.array([3, -0.5, 2, 7])
         self.y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8])
         self.metrics = RegressionExplainabilityMetrics()


    def test_all_explainability_metrics(self):
        for name in dir(self.metrics):
           if not name.startswith("_") and callable(getattr(self.metrics, name)):
                try:
                     if name == "custom_feature_importance_ratio":
                         result = getattr(self.metrics, name)(self.feature_importances_reg)
                     else:
                          result = getattr(self.metrics, name)(self.y_true_reg, self.y_pred_reg)
                     self.assertIsNotNone(result, f"Metric {name} returned None")

                except Exception as e:
                   self.fail(f"Metric {name} raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()