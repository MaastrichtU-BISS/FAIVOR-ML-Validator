import numpy as np

def mean_percentage_error(y_true, y_pred) -> float:
    """
    Calculates Mean Percentage Error for regression.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true target values.
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float
        The mean percentage error.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() == 0:
        return np.nan  # Avoid division by zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100