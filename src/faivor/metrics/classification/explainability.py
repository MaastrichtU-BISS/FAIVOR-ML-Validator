import numpy as np
from scipy.stats import entropy

def prediction_entropy(y_prob) -> float:
    """
    Calculates the entropy of predictions for classification.

    Entropy is a measure of uncertainty. Higher entropy in predictions indicates
    higher model uncertainty.  This function computes the average entropy across all predictions.

    Parameters
    ----------
    y_prob : array-like of shape (n_samples, n_classes) or (n_samples,)
        The predicted probabilities for each class. Can be either:
        - A 2D array of shape (n_samples, n_classes) where each row represents
          the probability distribution over classes for a single sample.
        - A 1D array of shape (n_samples,) for binary classification, representing
          the probability of the positive class (class 1).

    Returns
    -------
    float
        The average prediction entropy. Returns np.nan if input is empty or invalid.
    """
    y_prob = np.asarray(y_prob)
    if y_prob.size == 0:
        return np.nan

    if y_prob.ndim == 1: # assume binary classification and probabilities are for positive class
        y_prob = np.vstack([1 - y_prob, y_prob]).T # create 2D prob array: [[p(class0), p(class1)], ...]

    if np.any(y_prob < 0) or np.any(y_prob > 1):
        return np.nan # probabilities should be between 0 and 1

    # Normalize probabilities to ensure they sum to 1 (handle potential rounding errors)
    y_prob_normalized = y_prob / np.sum(y_prob, axis=1, keepdims=True)

    # Calculate entropy for each prediction
    entropies = entropy(y_prob_normalized, axis=1)

    return np.mean(entropies)