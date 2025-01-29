import pytest
import numpy as np
from sklearn import metrics as skm


from faivor.metrics.regression import metrics

# sample regression data
y_true_reg = np.array([3, 0.5, 2, 7, 4.2, 1])
y_pred_reg = np.array([2.5, 0.01, 2.1, 7.8, 3.9, 1.1])

def test_all_performance_metrics():
    sklearn_metrics_to_compare = {
        "mean_absolute_error": skm.mean_absolute_error,
        "mean_squared_error": skm.mean_squared_error,
        "r2_score": skm.r2_score,
    } # just a random assortment of metrics to compare to sklearn

    for metric in metrics.performance: # loop through all the performance metrics we loaded
        try:
            # Calculate the metric using the defined function
            if metric.function_name == "mean_percentage_error":
                result = metric.compute(y_true_reg, y_pred_reg)
            else:
                result = metric.compute(y_true_reg, y_pred_reg) # for most metrics, just compute with true and predicted values

            assert result is not None, f"Metric {metric.regular_name} returned None" # make sure we got a number back, not nothing

            # Compare with sklearn metric if applicable
            if metric.function_name in sklearn_metrics_to_compare: # if this metric is one we want to compare to sklearn
                sklearn_func = sklearn_metrics_to_compare[metric.function_name] # grab the sklearn function
                sklearn_result = sklearn_func(y_true_reg, y_pred_reg)
                assert np.allclose(result, sklearn_result), f"Metric {metric.regular_name} result does not match sklearn" # check if our result is basically the same as sklearn's

        except Exception as e:
            pytest.fail(f"Metric {metric.regular_name} raised an exception: {e}")