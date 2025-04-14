from sklearn import metrics as skm
import numpy as np

def error_rate(y_true, y_pred):
    """Calculates custom error rate for classification."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return 1 - skm.accuracy_score(y_true, y_pred)

def negative_predictive_value(y_true, y_pred) -> float:
    """
    Calculates negative predictive value.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true target values.
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float
        The negative predictive value.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp=skm.confusion_matrix(y_true,y_pred).ravel()
    return tn/(tn+fn)

def positive_predictive_value(y_true, y_pred) -> float:
    """
    Calculates positive predictive value for classification.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true target values.
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float
        The positive predictive value.
        """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp=skm.confusion_matrix(y_true,y_pred).ravel()
    return tp/(tp+fp)

def specificity(y_true, y_pred) -> float:
    """
    Calculates specificity for classification.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true target values.
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    float
        The specificity.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return skm.recall_score(y_true,y_pred,pos_label=0)