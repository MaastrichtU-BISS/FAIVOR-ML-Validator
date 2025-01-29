import pytest
import numpy as np

from faivor.metrics.classification.explainability import prediction_entropy

def test_prediction_entropy():
    y_prob = np.array([
        [0.1, 0.9],
        [0.5, 0.5],
        [0.9, 0.1],
        [0.3, 0.7]
    ])
    result = prediction_entropy(y_prob)
    assert result is not None, "Prediction entropy returned None"
    assert not np.isnan(result), "Prediction entropy should not return NaN for valid input"

    y_prob_empty = np.array([])
    result_empty = prediction_entropy(y_prob_empty)
    assert np.isnan(result_empty), "Prediction entropy with empty array should return NaN"

    y_prob_invalid = np.array([
        [0.1, 1.2], # invalid prob
        [0.5, 0.5]
    ])
    result_invalid = prediction_entropy(y_prob_invalid)
    assert np.isnan(result_invalid), "Prediction entropy with invalid probabilities should return NaN"

    y_prob_single_class = np.array([
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    result_single_class = prediction_entropy(y_prob_single_class)
    assert np.allclose(result_single_class, 0.0), "Prediction entropy with single class should be 0"

    y_prob_uniform = np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    result_uniform = prediction_entropy(y_prob_uniform)
    expected_uniform_entropy = - (0.5 * np.log(0.5) + 0.5 * np.log(0.5)) # entropy for [0.5, 0.5] using natural log
    assert np.allclose(result_uniform, expected_uniform_entropy), "Prediction entropy with uniform probabilities should be max entropy"

    y_prob_1d = np.array([0.1, 0.5, 0.9, 0.3]) # 1D probabilities, should be treated as prob of class 1
    result_1d = prediction_entropy(y_prob_1d)
    assert result_1d is not None, "Prediction entropy with 1D prob array should not return None"
    assert not np.isnan(result_1d), "Prediction entropy with 1D prob array should not return NaN"