import pytest
import numpy as np
import torch
from faivor.metrics.classification.explainability import (
    ClassificationExplainabilityMetrics,
)

metrics = ClassificationExplainabilityMetrics()

# Sample Classification Data
y_true_class = np.array([0, 1, 1, 0, 1, 0])
y_pred_class = np.array([0, 1, 0, 0, 1, 1])
probabilities_class = np.array(
    [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.7, 0.3], [0.2, 0.8]]
)


def test_all_explainability_metrics():
    for name in dir(metrics):
        if not name.startswith("_") and callable(getattr(metrics, name)):
            try:
                if name == "custom_prediction_entropy":
                    result = getattr(metrics, name)(probabilities_class)
                else:
                    result = getattr(metrics, name)(y_true_class, y_pred_class)
                assert result is not None, f"Metric {name} returned None"
            except Exception as e:
                pytest.fail(f"Metric {name} raised an exception: {e}")
