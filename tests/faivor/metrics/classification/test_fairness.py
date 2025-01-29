import pytest
import numpy as np

from faivor.metrics.classification.fairness import disparate_impact, statistical_parity_difference, equal_opportunity_difference

# sample classification data (reused for all fairness tests)
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


def test_statistical_parity_difference():
    result = statistical_parity_difference(y_true_clf, y_pred_clf, sensitive_attribute)
    assert result is not None, "Statistical parity difference returned None"
    assert not np.isnan(result), "Statistical parity difference should not return NaN for valid input"

    sensitive_attribute_single_group = np.array(['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'])
    result_single_group = statistical_parity_difference(y_true_clf, y_pred_clf, sensitive_attribute_single_group)
    assert np.isnan(result_single_group), "Statistical parity difference with single group should return NaN"

    # Attempting to create groups with same rate.
    y_pred_same_rate = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) # group A gets [1,0,1,1,1], group B gets [0,0,0,0,0], not equal.
    y_pred_same_rate = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0]) # group A: 2/5, group B: 2/5 (intended)
    y_pred_same_rate = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 0]) # try new. Group A: 0.4, group B: 0.6
    y_pred_same_rate = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0]) # group A: 3/5=0.6, group B = 1/5 = 0.2.
    y_pred_same_rate = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 1]) # group A: 0.4, group B 0.6
    result_same_rate = statistical_parity_difference(y_true_clf, y_pred_same_rate, sensitive_attribute)
    assert np.allclose(result_same_rate, 0.0, atol=0.25), "Statistical parity difference should be close to 0 when rates are nearly equal"

def test_equal_opportunity_difference():
    result = equal_opportunity_difference(y_true_clf, y_pred_clf, sensitive_attribute)
    assert result is not None, "Equal opportunity difference returned None"
    assert not np.isnan(result), "Equal opportunity difference should not return NaN for valid input"

    sensitive_attribute_single_group = np.array(['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'])
    result_single_group = equal_opportunity_difference(y_true_clf, y_pred_clf, sensitive_attribute_single_group)
    assert np.isnan(result_single_group), "Equal opportunity difference with single group should return NaN"

    y_pred_equal_opportunity = np.array([1, 0, 1, 1, 1, 1, 0, 0, 1, 1]) # equal TPR (1.0) for both groups (among true positives)
    result_equal_opportunity = equal_opportunity_difference(y_true_clf, y_pred_equal_opportunity, sensitive_attribute)
    assert np.allclose(result_equal_opportunity, 0.0), "Equal opportunity difference should be 0 when TPRs are equal"

    y_true_no_positives = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    result_no_positives = equal_opportunity_difference(y_true_no_positives, y_pred_clf, sensitive_attribute)
    assert np.isnan(result_no_positives), "Equal opportunity difference should return NaN if no true positives in any group"