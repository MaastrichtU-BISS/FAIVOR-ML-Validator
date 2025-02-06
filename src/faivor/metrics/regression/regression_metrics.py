from faivor.metrics.config_loader import load_metrics

PERFORMANCE_METRICS, FAIRNESS_METRICS, EXPLAINABILITY_METRICS = load_metrics("regression/regression_metrics.yaml")