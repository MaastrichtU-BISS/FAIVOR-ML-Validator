import numpy as np

def disparate_impact(y_true, y_pred, sensitive_attribute, favorable_outcome=1) -> float:
    """
    Calculates Disparate Impact for classification.

    Disparate Impact (DI) is the ratio of the rate of favorable outcomes for the
    disadvantaged group compared to the advantaged group. A common threshold for
    concern is DI < 0.8, indicating potential adverse impact.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true target values (binary: 0 or 1).
    y_pred : array-like of shape (n_samples,)
        The predicted target values (binary: 0 or 1).
    sensitive_attribute : array-like of shape (n_samples,)
        The sensitive attribute values (categorical).
    favorable_outcome : int or float, default=1
        The value representing the favorable outcome in y_true and y_pred.

    Returns
    -------
    float
        The disparate impact ratio. Returns np.nan if there's only one group or
        if the advantaged group has no favorable outcomes.
    """
    y_true, y_pred, sensitive_attribute = (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(sensitive_attribute),
    )

    unique_sensitive_values = np.unique(sensitive_attribute)
    if len(unique_sensitive_values) < 2:
        return np.nan  # Not applicable for less than 2 groups

    favorable_rates = {}
    for value in unique_sensitive_values:
        group_mask = sensitive_attribute == value
        group_size = group_mask.sum()
        if group_size == 0:
            favorable_rates[value] = 0 # Handle empty groups to avoid division by zero later, assume 0 favorable rate
        else:
            favorable_outcomes_count = np.sum(y_pred[group_mask] == favorable_outcome)
            favorable_rates[value] = favorable_outcomes_count / group_size

    rates = np.array(list(favorable_rates.values()))
    min_rate = np.min(rates)
    max_rate = np.max(rates)

    if max_rate == 0: # avoid division by zero if advantaged group has no favorable outcomes
        return np.nan

    return min_rate / max_rate