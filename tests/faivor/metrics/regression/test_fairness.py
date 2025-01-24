import pytest
import numpy as np
import torch
import torchmetrics as tm
from faivor.metrics.regression.regression_metrics import FAIRNESS_METRICS


# Sample Regression Data (Same as in original test.py, but smaller and more suitable for unit tests)
y_true_reg = np.array([3, -0.5, 2, 7, 4.2, 1, 9])
y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8, 3.9, 1.1, 8.5])
sensitive_attribute_reg = np.array([0, 1, 0, 1, 0, 1, 0])


def test_all_fairness_metrics():
    for metric in FAIRNESS_METRICS:
        try:
            # Calculate the metric using the defined function
            if metric.function_name == "demographic_parity_ratio":
                result = metric.compute(y_true_reg, y_pred_reg, sensitive_attribute=sensitive_attribute_reg)
            elif metric.function_name == "mean_absolute_error":
                # Convert to torch tensors for torchmetrics
                y_true_reg_torch = torch.tensor(y_true_reg)
                y_pred_reg_torch = torch.tensor(y_pred_reg)
                result = metric.compute(y_true_reg_torch, y_pred_reg_torch)
                # Use torchmetrics MeanAbsoluteError class or functional method
                expected_result = tm.functional.mean_absolute_error(y_pred_reg_torch, y_true_reg_torch)
                assert np.isclose(result, expected_result.item()), f"Metric {metric.regular_name} failed"
            else:
                result = metric.compute(y_true_reg, y_pred_reg)

            assert result is not None, f"Metric {metric.regular_name} returned None"
        except Exception as e:
            pytest.fail(f"Metric {metric.regular_name} raised an exception: {e}")
