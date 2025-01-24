from typing import List
from sklearn import metrics as skm
import numpy as np


def demographic_parity_ratio(y_true, y_pred, sensitive_attribute):
    """
    Calculates Demographic Parity Ratio for regression
    """
    y_true, y_pred, sensitive_attribute = (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(sensitive_attribute),
    )

    unique_sensitive_values = np.unique(sensitive_attribute)
    if len(unique_sensitive_values) < 2:
        return np.nan  # not applicable for less than 2 groups

    group_means = []
    for value in unique_sensitive_values:
        group_mask = sensitive_attribute == value
        if group_mask.sum() == 0:
            group_means.append(np.nan)  # to handle potential nan group mean
        else:
            group_means.append(np.mean(y_pred[group_mask]))

    group_means = np.asarray(group_means)
    if np.isnan(group_means).any():
        return np.nan  # to handle nan group means

    return np.min(group_means) / np.max(group_means)
