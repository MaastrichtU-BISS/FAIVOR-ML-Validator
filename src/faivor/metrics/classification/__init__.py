from .classification_metrics import (
    PERFORMANCE_METRICS as performance,
    FAIRNESS_METRICS as fairness,
    EXPLAINABILITY_METRICS as explainability
)

class ClassificationMetrics:
    def __init__(self):
        self.performance = performance
        self.fairness = fairness
        self.explainability = explainability

# Create an instance for easy access
metrics = ClassificationMetrics()