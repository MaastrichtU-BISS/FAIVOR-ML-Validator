import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, roc_auc_score

from faivor.metrics.confidence import bootstrap_ci


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    return y, p


def test_ci_brackets_point_estimate(clf_data):
    y, p = clf_data
    lower, upper = bootstrap_ci(roc_auc_score, y, p, stratify=True)
    assert lower < roc_auc_score(y, p) < upper


def test_ci_regression_metric():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    p = y + rng.normal(scale=0.5, size=200)
    lower, upper = bootstrap_ci(mean_squared_error, y, p)
    assert lower < mean_squared_error(y, p) < upper


def test_ci_is_deterministic_with_seed(clf_data):
    y, p = clf_data
    assert bootstrap_ci(roc_auc_score, y, p, seed=7) == bootstrap_ci(roc_auc_score, y, p, seed=7)


def test_higher_confidence_gives_wider_interval(clf_data):
    y, p = clf_data
    l90, u90 = bootstrap_ci(roc_auc_score, y, p, confidence=0.90)
    l99, u99 = bootstrap_ci(roc_auc_score, y, p, confidence=0.99)
    assert (u99 - l99) > (u90 - l90)


def test_small_sample_returns_none():
    assert bootstrap_ci(mean_squared_error, np.arange(5.0), np.arange(5.0)) is None


def test_always_failing_metric_returns_none(clf_data):
    y, p = clf_data

    def broken(a, b):
        raise ValueError("boom")

    assert bootstrap_ci(broken, y, p) is None


def test_single_class_unstratified_returns_none():
    y = np.ones(50)
    p = np.linspace(0, 1, 50)
    assert bootstrap_ci(roc_auc_score, y, p) is None
