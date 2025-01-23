import pytest
import numpy as np
import torch
from faivor.metrics.classification.fairness import ClassificationFairnessMetrics


# Sample Classification Data
y_true_class = np.array([0, 1, 1, 0, 1, 0, 1, 0])
y_pred_class = np.array([0, 1, 0, 0, 1, 1, 1, 0])
sensitive_attribute_class = np.array([0, 1, 0, 1, 0, 1, 0, 1])

metrics = ClassificationFairnessMetrics()

def test_all_fairness_metrics():
    for name in dir(metrics):
        if not name.startswith("_") and callable(getattr(metrics, name)):
            try:
                    if name == "custom_disparate_impact":
                        result = getattr(metrics, name)(y_true_class, y_pred_class, sensitive_attribute_class)
                    else:
                        result = getattr(metrics, name)(y_true_class, y_pred_class)
                    assert result is not None, f"Metric {name} returned None"
            except Exception as e:
                    pytest.fail(f"Metric {name} raised an exception: {e}")