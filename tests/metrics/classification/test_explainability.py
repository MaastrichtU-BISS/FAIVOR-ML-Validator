import pytest
import numpy as np

from faivor.metrics.classification.explainability import prediction_entropy, confidence_score, margin_of_confidence

# Sample probability data (reused for all tests)
y_prob = np.array([
    [0.1, 0.9], # confident prediction
    [0.5, 0.5], # uncertain prediction
    [0.9, 0.1], # confident prediction
    [0.3, 0.7]  # moderately confident prediction
])
y_prob_1d = np.array([0.1, 0.5, 0.9, 0.3]) # 1D probabilities for binary case


def test_prediction_entropy():
    result = prediction_entropy(y_prob)
    assert result is not None, "Prediction entropy returned None"
    assert not np.isnan(result), "Prediction entropy should not return NaN for valid input"

    result_1d = prediction_entropy(y_prob_1d)
    assert result_1d is not None, "Prediction entropy with 1D prob array should not return None"
    assert not np.isnan(result_1d), "Prediction entropy with 1D prob array should not return NaN"

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


def test_confidence_score():
    result = confidence_score(y_prob)
    assert result is not None, "Confidence score returned None"
    assert not np.isnan(result), "Confidence score should not return NaN for valid input"

    result_1d = confidence_score(y_prob_1d)
    assert result_1d is not None, "Confidence score with 1D prob array should not return None"
    assert not np.isnan(result_1d), "Confidence score with 1D prob array should not return NaN"

    y_prob_empty = np.array([])
    result_empty = confidence_score(y_prob_empty)
    assert np.isnan(result_empty), "Confidence score with empty array should return NaN"

    y_prob_invalid = np.array([
        [0.1, 1.2], # invalid prob
        [0.5, 0.5]
    ])
    result_invalid = confidence_score(y_prob_invalid)
    assert np.isnan(result_invalid), "Confidence score with invalid probabilities should return NaN"

    expected_confidence = np.mean([0.9, 0.5, 0.9, 0.7]) # average of max probabilities
    assert np.allclose(result, expected_confidence), "Confidence score calculation incorrect"


def test_margin_of_confidence():
    result = margin_of_confidence(y_prob)
    assert result is not None, "Margin of confidence returned None"
    assert not np.isnan(result), "Margin of confidence should not return NaN for valid input for binary case"

    result_1d = margin_of_confidence(y_prob_1d)
    assert result_1d is not None, "Margin of confidence with 1D prob array should not return None"
    assert not np.isnan(result_1d), "Margin of confidence with 1D prob array should not return NaN"

    y_prob_empty = np.array([])
    result_empty = margin_of_confidence(y_prob_empty)
    assert np.isnan(result_empty), "Margin of confidence with empty array should return NaN"

    y_prob_invalid = np.array([
        [0.1, 1.2], # invalid prob
        [0.5, 0.5]
    ])
    result_invalid = margin_of_confidence(y_prob_invalid)
    assert np.isnan(result_invalid), "Margin of confidence with invalid probabilities should return NaN"

    y_prob_multiclass = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]) # multiclass
    result_multiclass = margin_of_confidence(y_prob_multiclass)
    assert np.isnan(result_multiclass), "Margin of confidence with multiclass should return NaN"

    expected_margin = np.mean([np.abs(0.9 - 0.1), np.abs(0.5 - 0.5), np.abs(0.1 - 0.9), np.abs(0.7 - 0.3)]) # average of margins
    assert np.allclose(result, expected_margin), "Margin of confidence calculation incorrect"