import pytest
import numpy as np
from sklearn import metrics as skm

from faivor.metrics.classification import metrics

# sample classification data
y_true_clf = np.array([1, 0, 1, 1, 0, 1])
y_pred_clf = np.array([1, 1, 0, 1, 0, 1])
y_prob_clf = np.array([
    [0.1, 0.9],
    [0.8, 0.2],
    [0.3, 0.7],
    [0.2, 0.8],
    [0.6, 0.4],
    [0.4, 0.6]
]) # probabilities for binary classification

def test_all_performance_metrics():
    sklearn_metrics_to_compare = {
        "accuracy_score": skm.accuracy_score,
        "f1_score": skm.f1_score,
        "precision_score": skm.precision_score,
        "recall_score": skm.recall_score,
        "roc_auc_score": skm.roc_auc_score,
        "log_loss": skm.log_loss,
        "balanced_accuracy_score": skm.balanced_accuracy_score,
        "top_k_accuracy_score": skm.top_k_accuracy_score # added for comparison
    } # just a random assortment of metrics to compare to sklearn

    for metric in metrics.performance: # loop through all the performance metrics we loaded
        try:
            # Calculate the metric using the defined function
            if metric.function_name in ["roc_auc_score", "average_precision_score", "log_loss", "brier_score_loss", "top_k_accuracy_score"]: # added top_k_accuracy_score here
                result = metric.compute(y_true_clf, y_prob_clf[:, 1]) # these need probability scores, use probabilities of positive class for binary
            elif metric.function_name in ["top_k_accuracy_score"]: # redundant condition, but kept for clarity - now handled in the above condition
                result = metric.compute(y_true_clf, y_prob_clf, k=2) # example for top_k, k needs to be passed - NOT NEEDED ANYMORE FOR BINARY CASE WITH 1D PROBS
            elif metric.function_name in ["roc_curve", "precision_recall_curve", "confusion_matrix"]:
                result = metric.compute(y_true_clf, y_pred_clf) # these return arrays or matrices, not single values, still test no error
                assert result is not None
                continue # no numerical comparison for these
            else:
                result = metric.compute(y_true_clf, y_pred_clf) # for most metrics, just compute with true and predicted values

            assert result is not None, f"Metric {metric.regular_name} returned None" # make sure we got a number back, not nothing

            # Compare with sklearn metric if applicable
            if metric.function_name in sklearn_metrics_to_compare: # if this metric is one we want to compare to sklearn
                sklearn_func = sklearn_metrics_to_compare[metric.function_name] # grab the sklearn function
                if metric.function_name in ["roc_auc_score", "log_loss", "average_precision_score", "top_k_accuracy_score"]: # added top_k_accuracy_score here
                    sklearn_result = sklearn_func(y_true_clf, y_prob_clf[:, 1]) # use 1D probs for sklearn too
                else:
                    sklearn_result = sklearn_func(y_true_clf, y_pred_clf)
                assert np.allclose(result, sklearn_result, atol=1e-5), f"Metric {metric.regular_name} result does not match sklearn" # check if our result is basically the same as sklearn's

        except Exception as e:
            pytest.fail(f"Metric {metric.regular_name} raised an exception: {e}")