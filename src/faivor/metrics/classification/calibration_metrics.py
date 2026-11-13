from sklearn.linear_model import LogisticRegression
import numpy as np
from calzone.utils import reliability_diagram
from calzone.metrics import calculate_ece_mce, lowess_regression_analysis
"""
    Script to calculate calibration metrics as part of validation procedure (not model update)
    
    Adapted functions from the code from Tuur Leeuwenberg https://github.com/tuur/CalibrationSlopeInterceptPython/blob/main/CalibrationSlopeIntercept.py 
    with following decsription: 
    Python implementations of calibration slope, calibration intercept, and calibration in the large for prediction models with a binary outcome.

    Reference (for explanation about the measures): Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, Steyerberg EW. A calibration hierarchy for risk models was defined: from utopia to empirical data. Journal of clinical epidemiology. 2016 Jun 1;74:167-76.
    # Disclaimer:
    # -----------
    # This code is not affiliated (nor checked by) the authors of the reference above. I did verify on multiple datasets that the outputs are the same as for the val.prob function in the 'rms' R-package using the test() function below.

    """
def logit_function(p):
    p = np.clip(p, 0 + (1/np.inf), 1-(1/np.inf)) # to prevent division by 0
    return np.log(p/(1-p))

def calibration_slope_intercept_inthelarge(y_true, y_pred):
    """
     A function to calculate calibration measures for binary outcomes.
    :param y_pred: A list/vector of predicted probabilities of the outcome
    :param y_true: A list/vector of the true (binary) outcome.
    :return: A tuple of the (calibration slope, calibration intercept, and the calibration-in-the-large)
    """
    y_pred_linear_scores = logit_function(y_pred)
    calib_model = LogisticRegression(penalty=None)
    calib_model.fit(X=np.array(y_pred_linear_scores).reshape(-1, 1), y=np.array(y_true))
    return (calib_model.coef_[0][0], calib_model.intercept_[0])  # , CITL)

def calibration_intercept(y_true, y_pred) -> float:
    """
        Calculates intercept of the calibration curve.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The true target values.
        y_pred : array-like of shape (n_samples,)

        Returns
        -------
        float
            The calibration intercept.
        """

    slope, intercept = calibration_slope_intercept_inthelarge(y_true, y_pred)
    return intercept

def calibration_slope(y_true, y_pred) -> float:
    """
        Calculates slope of the calibration curve.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The true target values.
        y_pred : array-like of shape (n_samples,)

        Returns
        -------
        float
            The calibration slope.
        """

    slope, intercept = calibration_slope_intercept_inthelarge(y_true, y_pred)
    return slope

# TODO: add uncertainity-calibration from https://pypi.org/project/uncertainty-calibration/#description to calculate ECI, ICI, ECE

def O_E_ratio(y_true, y_pred) -> float:
    """
            Calculates the ratio between observed and expected outcome.
            Adapted function from https://github.com/Oefuli/samd-model-evaluation/tree/main

            Parameters
            ----------
            y_true : array-like of shape (n_samples,)
                The true target values.
            y_pred : array-like of shape (n_samples,)

            Returns
            -------
            float
                The O:E ratio.
            """

    obs, exp = np.sum(y_true), np.sum(y_pred)
    return float(obs / exp) if exp > 0 else np.nan

def calibration_ECE(y_true, y_pred) -> float:
    """
        Calculates the ECE - Expected Calibration Error.
        Based on the calzone tool. https://calzone-docs.readthedocs.io/en/latest/notebooks/ece_mce.html

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The true target values.
        y_pred : array-like of shape (n_samples,)

        Returns
        -------
        float
            ECE.
        """

    reliability, confidence, bin_edge, bin_count = reliability_diagram(
        y_true=y_true,
        y_proba=y_pred,
        num_bins=15,
        class_to_plot=None,
        save_path=None,
    )

    ece_h_top_class, mce_h_top_class = calculate_ece_mce(reliability, confidence, bin_counts=bin_count)

    return ece_h_top_class

def calibration_ICI(y_true, y_pred) -> float:
    """
            Calculates the ICI - Integrated Calibration Index.
            Based on calzone tool https://calzone-docs.readthedocs.io/en/latest/notebooks/ici.html

            Parameters
            ----------
            y_true : array-like of shape (n_samples,)
                The true target values.
            y_pred : array-like of shape (n_samples,)

            Returns
            -------
            float
                ICI.
            """
    loess_ici, lowess_fit_p, lowess_fit_p_correct = lowess_regression_analysis(y_true,
                                                                               y_pred,
                                                                               class_to_calculate=1, span=0.5,
                                                                               delta=0.001, it=0)
    return loess_ici
