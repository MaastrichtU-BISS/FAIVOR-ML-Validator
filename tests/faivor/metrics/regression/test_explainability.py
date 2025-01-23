import pytest
import numpy as np
import torch
from faivor.metrics.regression.explainability import RegressionExplainabilityMetrics

# Sample Regression Data
feature_importances_reg = np.array([0.1, 0.2, 0.7, 0.05, 0.05])
y_true_reg = np.array([3, -0.5, 2, 7])
y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8])
metrics = RegressionExplainabilityMetrics()


def test_all_explainability_metrics():
    for name in dir(metrics):
        if not name.startswith("_") and callable(getattr(metrics, name)):
            try:
                    if name == "custom_feature_importance_ratio":
                        result = getattr(metrics, name)(feature_importances_reg)
                    else:
                        result = getattr(metrics, name)(y_true_reg, y_pred_reg)
                    assert result is not None, f"Metric {name} returned None"

            except Exception as e:
                pytest.fail(f"Metric {name} raised an exception: {e}")