import logging
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_ci(
    metric_fn: Callable,
    y_true,
    y_pred,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = 42,
    stratify: bool = False,
    min_valid_fraction: float = 0.5,
    min_samples: int = 10,
) -> Optional[Tuple[float, float]]:
    """
    Percentile bootstrap confidence interval for any metric(y_true, y_pred).

    Parameters
    ----------
    metric_fn : callable
        Function taking (y_true, y_pred) and returning a scalar.
    n_boot : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (0.95 -> 2.5th and 97.5th percentiles).
    seed : int, optional
        Seed for reproducibility.
    stratify : bool
        Resample within each class of y_true (use for classification so that
        resamples never contain a single class).
    min_samples : int
        Return None for samples smaller than this (unstable intervals, e.g. small subgroups).
    min_valid_fraction : float
        Return None if fewer than this fraction of resamples produced a finite value.

    Returns
    -------
    (lower, upper) or None if the interval cannot be estimated.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n < max(2, min_samples):
        return None

    rng = np.random.default_rng(seed)
    if stratify:
        class_idx = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]

    values = []
    for _ in range(n_boot):
        if stratify:
            idx = np.concatenate([rng.choice(ci, size=len(ci), replace=True) for ci in class_idx])
        else:
            idx = rng.integers(0, n, size=n)
        try:
            v = float(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
        if np.isfinite(v):
            values.append(v)

    if len(values) < max(2, int(min_valid_fraction * n_boot)):
        logger.debug("Too few valid bootstrap resamples (%d/%d)", len(values), n_boot)
        return None

    alpha = 1.0 - confidence
    lower, upper = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lower), float(upper)
