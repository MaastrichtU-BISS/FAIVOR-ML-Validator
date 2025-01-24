from sklearn import metrics as skm
from typing import Callable, List
import torchmetrics as tm

from faivor.metrics.metric import ModelMetric
from faivor.metrics.regression import performance as performance_matrics
from faivor.metrics.regression import fairness as fairness_metrics
from faivor.metrics.regression import explainability as explainability_metrics


PERFORMANCE_METRICS: List[ModelMetric] = [
    ModelMetric(
        function_name="mean_absolute_error",
        regular_name="Mean Absolute Error",
        description="The mean of the absolute errors between true and predicted values.",
        func=skm.mean_absolute_error,
    ),
    ModelMetric(
        function_name="mean_squared_error",
        regular_name="Mean Squared Error",
        description="The mean of the squared errors between true and predicted values.",
        func=skm.mean_squared_error,
    ),
    ModelMetric(
        function_name="mean_squared_log_error",
        regular_name="Mean Squared Logarithmic Error",
        description="Regression loss using the log of true and predicted values.",
        func=skm.mean_squared_log_error,
    ),
    ModelMetric(
        function_name="median_absolute_error",
        regular_name="Median Absolute Error",
        description="The median of the absolute errors between true and predicted values.",
        func=skm.median_absolute_error,
    ),
    ModelMetric(
        function_name="r2_score",
        regular_name="R² Score",
        description="The coefficient of determination regression score.",
        func=skm.r2_score,
    ),
    ModelMetric(
        function_name="explained_variance_score",
        regular_name="Explained Variance Score",
        description="Measures the proportion of variance explained by the model.",
        func=skm.explained_variance_score,
    ),
    ModelMetric(
        function_name="max_error",
        regular_name="Max Error",
        description="The maximum absolute difference between true and predicted values.",
        func=skm.max_error,
    ),
    ModelMetric(
        function_name="mean_poisson_deviance",
        regular_name="Mean Poisson Deviance",
        description="Mean Poisson deviance regression loss.",
        func=skm.mean_poisson_deviance,
    ),
    ModelMetric(
        function_name="mean_gamma_deviance",
        regular_name="Mean Gamma Deviance",
        description="Mean gamma deviance regression loss.",
        func=skm.mean_gamma_deviance,
    ),
    ModelMetric(
        function_name="d2_absolute_error_score",
        regular_name="D² Absolute Error Score",
        description="The proportion of variance explained using absolute errors.",
        func=skm.d2_absolute_error_score,
    ),
    ModelMetric(
        function_name="mean_pinball_loss",
        regular_name="Mean Pinball Loss",
        description="The mean pinball loss for quantile regression.",
        func=skm.mean_pinball_loss,
    ),
    ModelMetric(
        function_name="mean_percentage_error",
        regular_name="Mean Percentage Error",
        description="Calculates the mean percentage error for regression, ignoring zero true values.",
        func=performance_matrics.mean_percentage_error,
    ),
]

FAIRNESS_METRICS: List[ModelMetric] = [
    ModelMetric(
        function_name="mean_absolute_error",
        regular_name="Mean Absolute Error (Torch)",
        description="The mean of the absolute errors using Torch for regression fairness evaluation.",
        func=tm.MeanAbsoluteError,
        is_torch=True,
    ),
    ModelMetric(
        function_name="mean_squared_error",
        regular_name="Mean Squared Error (Torch)",
        description="The mean of the squared errors using Torch for regression fairness evaluation.",
        func=tm.MeanSquaredError,
        is_torch=True,
    ),
    ModelMetric(
        function_name="mean_absolute_percentage_error",
        regular_name="Mean Absolute Percentage Error (Torch)",
        description="The mean absolute percentage error using Torch for regression fairness evaluation.",
        func=tm.MeanAbsolutePercentageError,
        is_torch=True,
    ),
    ModelMetric(
        function_name="demographic_parity_ratio",
        regular_name="Custom Demographic Parity Ratio",
        description=(
            "Calculates the demographic parity ratio for regression by comparing the average predicted values across different sensitive attribute groups."
        ),
        func=fairness_metrics.demographic_parity_ratio,
    ),
]


EXPLAINABILITY_METRICS: List[ModelMetric] = [
    ModelMetric(
        function_name="feature_importance_ratio",
        regular_name="Feature Importance Ratio",
        description="Calculates the ratio of feature importance for regression explainability.",
        func=explainability_metrics.feature_importance_ratio,
    ),
]
