import pytest
import numpy as np
from sklearn import metrics as skm

from faivor.metrics.regression.regression_metrics import PERFORMANCE_METRICS


# Sample Regression Data
y_true_reg = np.array([3, 0.5, 2, 7, 4.2, 1])
y_pred_reg = np.array([2.5, 0.01, 2.1, 7.8, 3.9, 1.1])

def test_all_performance_metrics():
    for metric in PERFORMANCE_METRICS:
        try:
            # Calculate the metric using the defined function
            if metric.function_name == "mean_percentage_error":
                result = metric.compute(y_true_reg, y_pred_reg)
            elif metric.function_name == "mean_absolute_error":
                result = metric.compute(y_true_reg, y_pred_reg)
                assert np.allclose(result, skm.mean_absolute_error(y_true_reg, y_pred_reg)), f"Metric {metric.regular_name} failed"
            else:
                result = metric.compute(y_true_reg, y_pred_reg)

            assert result is not None, f"Metric {metric.regular_name} returned None"
        except Exception as e:
            pytest.fail(f"Metric {metric.regular_name} raised an exception: {e}")

