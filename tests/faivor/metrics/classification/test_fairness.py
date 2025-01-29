import pytest
import numpy as np

from faivor.metrics.classification.fairness import disparate_impact

# sample classification data
y_true_clf = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
y_pred_clf = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
sensitive_attribute = np.array(['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'B'])

def test_disparate_impact():
    result = disparate_impact(y_true_clf, y_pred_clf, sensitive_attribute)
    assert result is not None, "Disparate impact returned None"
    assert not np.isnan(result), "Disparate impact should not return NaN for valid input"

    sensitive_attribute_single_group = np.array(['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'])
    result_single_group = disparate_impact(y_true_clf, y_pred_clf, sensitive_attribute_single_group)
    assert np.isnan(result_single_group), "Disparate impact with single group should return NaN"

    y_pred_no_favorable = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # no favorable outcomes predicted
    result_no_favorable = disparate_impact(y_true_clf, y_pred_no_favorable, sensitive_attribute)
    assert np.isnan(result_no_favorable), "Disparate impact should return NaN if no favorable outcomes in advantaged group"

    sensitive_attribute_empty_group = np.array(['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'C']) # group C is empty
    result_empty_group = disparate_impact(y_true_clf, y_pred_clf, sensitive_attribute_empty_group)
    assert not np.isnan(result_empty_group), "Disparate impact with empty group in sensitive attribute should not return NaN if there are other groups"

    y_pred_all_favorable = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    result_all_favorable = disparate_impact(y_true_clf, y_pred_all_favorable, sensitive_attribute)
    assert np.allclose(result_all_favorable, 1.0), "Disparate impact should be 1.0 when all groups have 100% favorable outcome rate"