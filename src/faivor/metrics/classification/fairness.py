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


def statistical_parity_difference(y_true, y_pred, sensitive_attribute, favorable_outcome=1) -> float:
    """
    Calculates Statistical Parity Difference for classification.

    Statistical Parity Difference (SPD) is the difference in the rate of favorable outcomes
    between the least advantaged and the most advantaged group. Ideally, SPD should be close to 0.

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
        The statistical parity difference. Returns np.nan if there's only one group.
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
            favorable_rates[value] = 0 # Handle empty groups, assume 0 favorable rate
        else:
            favorable_outcomes_count = np.sum(y_pred[group_mask] == favorable_outcome)
            favorable_rates[value] = favorable_outcomes_count / group_size

    rates = np.array(list(favorable_rates.values()))
    return np.max(rates) - np.min(rates)


def equal_opportunity_difference(y_true, y_pred, sensitive_attribute, favorable_outcome=1) -> float:
    """
    Calculates Equal Opportunity Difference for classification.

    Equal Opportunity Difference (EOD) is the difference in true positive rates between groups.
    It aims to ensure that individuals who truly deserve the favorable outcome have an equal chance
    of receiving it, regardless of their sensitive attribute. Ideally, EOD should be close to 0.

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
        The equal opportunity difference. Returns np.nan if there's only one group
        or if there are no true positive cases in *any* group for the favorable outcome.
    """
    y_true, y_pred, sensitive_attribute = (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(sensitive_attribute),
    )

    unique_sensitive_values = np.unique(sensitive_attribute)
    if len(unique_sensitive_values) < 2:
        return np.nan  # Not applicable for less than 2 groups

    true_positive_rates = {}
    total_true_positives = 0 # Track total true positives across all groups for favorable outcome
    for value in unique_sensitive_values:
        group_mask = sensitive_attribute == value
        y_true_group = y_true[group_mask]
        y_pred_group = y_pred[group_mask]

        tp = np.sum((y_true_group == favorable_outcome) & (y_pred_group == favorable_outcome))
        actual_positives = np.sum(y_true_group == favorable_outcome)
        total_true_positives += actual_positives # accumulate total true positives

        if actual_positives == 0:
            true_positive_rates[value] = 0 # avoid division by zero if no true positives in group
        else:
            true_positive_rates[value] = tp / actual_positives

    if total_true_positives == 0: # if no true positives for favorable outcome in *any* group, return NaN
        return np.nan

    rates = np.array(list(true_positive_rates.values()))
    return np.max(rates) - np.min(rates)