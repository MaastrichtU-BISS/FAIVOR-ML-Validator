from sklearn import metrics as skm
import numpy as np

def error_rate(y_true, y_pred):
    """Calculates custom error rate for classification."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return 1 - skm.accuracy_score(y_true, y_pred)

