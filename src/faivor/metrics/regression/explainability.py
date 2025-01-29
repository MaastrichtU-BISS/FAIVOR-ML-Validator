import numpy as np

def feature_importance_ratio(feature_importances) -> float:
    """
    Calculate a ratio to assess feature importance

    Parameters
    ----------
    feature_importances : array-like of shape (n_features,)
        The feature importances.

    Returns
    -------
    float
        The feature importance ratio.
    """
    feature_importances = np.asarray(feature_importances)
    if len(feature_importances) == 0:
        return np.nan
    return np.min(feature_importances) / np.max(feature_importances)