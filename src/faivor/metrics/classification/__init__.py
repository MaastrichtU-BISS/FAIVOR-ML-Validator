from .classification_metrics import (
    OVERALL_PERFORMANCE_METRICS as overall_performance,
    FAIRNESS_METRICS as fairness,
    EXPLAINABILITY_METRICS as explainability,
    DISCRIMINATION_METRICS as discrimination,
    CALIBRATION_METRICS as calibration,
    CLASSIFICATION_METRICS as classification,
    CLINICAL_UTILITY_METRICS as clinical_utility,
)

class ClassificationMetrics:
    def __init__(self):
        self.overall_performance = overall_performance
        self.discrimination = discrimination
        self.clinical_utility = clinical_utility
        self.classification = classification
        self.calibration = calibration
        self.fairness = fairness
        self.explainability = explainability

# Create an instance for easy access
metrics = ClassificationMetrics()