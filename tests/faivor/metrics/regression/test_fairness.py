import pytest
import numpy as np

from faivor.metrics.regression.fairness import demographic_parity_ratio

# sample regression data
y_true_reg = np.array([3, 0.5, 2, 7, 4.2, 1])
y_pred_reg = np.array([2.5, 0.01, 2.1, 7.8, 3.9, 1.1])

def test_demographic_parity_ratio():
    sensitive_attribute = np.array(['A', 'A', 'B', 'B', 'A', 'B'])
    result = demographic_parity_ratio(y_true_reg, y_pred_reg, sensitive_attribute)
    assert result is not None, "Demographic parity ratio returned None"

    sensitive_attribute_single_group = np.array(['A', 'A', 'A', 'A', 'A', 'A'])
    result_single_group = demographic_parity_ratio(y_true_reg, y_pred_reg, sensitive_attribute_single_group)
    assert np.isnan(result_single_group), "Demographic parity ratio with single group should return NaN"

    sensitive_attribute_empty_group = np.array(['A', 'A', 'B', 'B', 'A', 'C']) # group C has no samples in y_pred
    result_empty_group = demographic_parity_ratio(y_true_reg, y_pred_reg, sensitive_attribute_empty_group)
    assert not np.isnan(result_empty_group), "Demographic parity ratio with empty group in sensitive attribute should not return NaN if there are other groups"

    y_pred_all_nan = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    result_nan_pred = demographic_parity_ratio(y_true_reg, y_pred_all_nan, sensitive_attribute)
    assert np.isnan(result_nan_pred), "Demographic parity ratio with nan predictions should return NaN"