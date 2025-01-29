import pytest
import numpy as np

from faivor.metrics.regression.explainability import feature_importance_ratio

def test_feature_importance_ratio():
    feature_importances = np.array([0.1, 0.2, 0.5, 0.2])
    result = feature_importance_ratio(feature_importances)
    assert result is not None, "Feature importance ratio returned None"

    feature_importances_empty = np.array([])
    result_empty = feature_importance_ratio(feature_importances_empty)
    assert np.isnan(result_empty), "Feature importance ratio with empty array should return NaN"

    feature_importances_single = np.array([0.5])
    result_single = feature_importance_ratio(feature_importances_single)
    assert result_single == 1.0, "Feature importance ratio with single value should return 1.0"

    feature_importances_equal = np.array([0.3, 0.3, 0.3])
    result_equal = feature_importance_ratio(feature_importances_equal)
    assert result_equal == 1.0, "Feature importance ratio with equal values should return 1.0"