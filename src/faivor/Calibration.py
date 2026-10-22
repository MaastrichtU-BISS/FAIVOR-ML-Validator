import logging
from typing import  Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


class Calibration:
   """
   Implements original and recalibrated logistic models (m0-m3)
   Integrates with FAVOR-ML-Validator pipeline to:
   1. Accept predictions from containerized model
   2. Fit update methods (m1-m3) on those predictions
   3. Model comparison metrics (log-likelihood, AUC)
   4. Select best calibration model using closed testing framework

   Update methods:
   m0 - original model ( no update, predictors used as they are)
   m1 - intercept re-estimated only (calibration-in-the-large)
   m2 - intercept and slope are re-estimated (logistic recalibration)
   m3 - all coefficients are re-estimated using the original predictors
   """
   def __init__(self,
                predictions: List[float],                  #list of predicted probabilities
                expected_outputs: List[Dict[str, Any]],    #list of dict with true outcome
                outcome_col: str,                          #key name of outcome inside the dict
                input_data:Optional[pd.DataFrame] = None   #optional predictor table, None if not given
                ) -> None:
      """
       Store predictions, outcomes and predictor data ( optional) in one table
       Parameters:
       predictions: List[float]
          Predicted probabilities returned from the docker model
       expected_outputs: List[Dict[str, Any]]
         Expected outcomes containing the true outcomes
       outcome_col: str
         Key to look up the outcome inside each record
       input_data: pd.DataFrame
         Predictor values for each patient (needed for m3 only)
      """
      #Input checks
      #Stop and throw error if no predictions were given (empty list)
      if not predictions:
          raise ValueError("predictions is empty")
      #Stop and throw error if no true outcomes to compare against
      if expected_outputs is None or len(expected_outputs) == 0:
          raise ValueError("expected_outputs is empty")
      #Every patient record must have one prediction and one outcome,
      #so both lists should be of equal length
      if len(predictions) != len(expected_outputs):
          raise ValueError("predictions and expected_outputs must have the same length")

      #Store the value on object so that other methods can use it later
      self.predictions = predictions
      self.expected_outputs = expected_outputs
      self.outcome_col = outcome_col

      #Create empty table, all later methods use this table
      self.data = pd.DataFrame()
      #Add predictions to table with column name "predictions"
      self.data["predictions"] = predictions

     #Get the true outcome of each patient's record
      outcome_vals = []
      for outcome in expected_outputs:
          #reads the value only if the record is a dict and has the outcome key
          if isinstance(outcome, dict) and outcome_col in outcome:
              try:
                  #converts to a number and adds it to list
                  outcome_vals.append(float(outcome[outcome_col]))
              except (ValueError,TypeError):
                  # in case it can't be converted to a number, store as NaN
                  outcome_vals.append(np.nan)
          else:
              #record is not a dict or has no outcome key, store as NaN
              outcome_vals.append(np.nan)
      #Store the outcomes in a new column, named as "outcome_col"
      self.data[outcome_col] = outcome_vals

      #Attach predictor columns (skip any name already in table
      #method needed for m3 to refit the original predictors
      #Runs only if predictor data is given
      if input_data is not None:
          input_df = input_data.copy() if isinstance(input_data, pd.DataFrame) else pd.DataFrame(input_data)
          #Go through each predictor column one by one
          for col in input_df.columns:
              #skip column if a column with same name already exists
              if col not in self.data.columns:
                  self.data[col] = input_df[col].values
      #To store the models once the fitting methods are run
      self._fitted_m1 = None
      self._fitted_m2 = None
      self._fitted_m3 = None


   @classmethod
   def from_docker_output(cls,
                          docker_result: Dict[str, Any],     #Dict returned by execute_model() (output of container run), holds predictions
                          expected_outputs: List[Dict[str, Any]], #true outcomes
                          outcome_col:str,                          #name of the outcome key
                          input_data:Optional[pd.DataFrame] = None) -> 'Calibration': #return calibration instance
       """
       Alternative constructor that reads predictions straight from dictionary returned by execute_model()
       """
       #Docker result must be a dictionary, otherwise stops and throws an error
       if not isinstance(docker_result, dict):
           raise ValueError("docker_result must be a dictionary")
       #The dictionary must contain a "predictions" key, otherwise stops and throws an error
       if "predictions" not in docker_result:
           raise ValueError("docker_result must contain 'predictions' key")
       #Take list of predictions from the dictionary
       predictions = docker_result["predictions"]

       #If the prediction list is empty, stop and throw an error
       if predictions is None or len(predictions) == 0:
           raise ValueError("predictions is empty")

       #return calibration instance with predictions pulled from the docker result
       return cls(predictions=predictions,
                  expected_outputs=expected_outputs,
                  outcome_col=outcome_col,
                  input_data=input_data)

   def drop_missing_predictions(self) -> None:
       """
       Drops rows with missing data in predictions.
       """
       self.data = self.data.dropna(subset=["predictions",self.outcome_col]).copy()

   def compute_linear_predictor(self) -> np.ndarray:
       """
       Compute linear predictor from predictions using logit transformation.
       Serves as baseline for calibration models
       Returns linear predictor values(log odds)
       """

       #Keep predictions in valid probability range, strictly between 0 and 1
       predictions = np.clip(self.data["predictions"].values, 1e-15, 1-1e-15)
       #Convert probabilities to log-odds
       lp = np.log(predictions / (1- predictions))
       self.data["linear_predictor"] = lp
       return lp

   def predict_m0(self) -> pd.Series:
       """
       Get predicted probabilities for m0 (original model predictions)
       """
       self.data["m0_pred"] = self.data["predictions"].values
       return self.data["m0_pred"]

   def fit_m1_recalibration_in_the_large(self) -> pd.Series:
       """
       Fit m1: recalibration in the large (adjust intercept only)
       """
       #Check if linear predictor is computed
       if "linear_predictor" not in self.data.columns:
          self.compute_linear_predictor()
        #Fit a logistic regression model on outcome of linear predictor using glm function
       model = smf.glm(
          formula = f" {self.outcome_col} ~ 1", #formula estimates only an intercept, no other variables
                                                #fixed coefficient at 1, so slope cannot change
          data = self.data,
          family = sm.families.Binomial(),
          offset = self.data["linear_predictor"], #adds original model's logit to every patient
      ).fit(disp=0)

       #Save fitted model
       self._fitted_m1 = model
       #Get new estimated intercept
       intercept = model.params["Intercept"]
       #New logit for each patient ( adjusted intercept + original logit)
       lp = intercept + self.data["linear_predictor"]
       #Convert new logit back to probability and store it as new column, m1_pred
       self.data["m1_pred"] = 1 / (1 + np.exp(-lp))
       #Return m1 predicted probabilities
       return self.data["m1_pred"]

   def fit_m2_logistic_calibration(self) -> pd.Series:
       """
       Fit m2: logistic calibration (estimate intercept and slope)
       """
       #check if linear predictor is computed
       if "linear_predictor" not in self.data.columns:
           self.compute_linear_predictor()
      #Fit logistic regression model on outcome of linear predictor using glm function
       model = smf.glm(
           formula = f" {self.outcome_col} ~ linear_predictor", #formula estimates intercept and slope
           data = self.data,
           family = sm.families.Binomial(),
       ).fit(disp=0)

       #Save fitted model
       self._fitted_m2 = model
       #Store predicted probabilities after refitting model
       self.data["m2_pred"] = model.fittedvalues
       #Return m2 predicted probabilities
       return self.data["m2_pred"]

   def _resolve_covariate_columns(self, covariate_columns: Optional[List[str]] = None) -> List[str]:
       """
       Return the given covariate columns, or default to all numeric predictor columns in the table
       """
       if covariate_columns is None:
           #Exclude columns that are not predictors
           exclude_columns = {"predictions", "m0_pred", "m1_pred", "m2_pred", "m3_pred", self.outcome_col,"linear_predictor"}

           #Keep every numeric columns in the tabel that is not in the exclude list and store as covariates
           covariate_columns = [col for col in self.data.select_dtypes(include=[np.number]).columns if col not in exclude_columns]
       return covariate_columns

   def fit_m3_model_revision(self, covariate_columns: Optional[List[str]] = None) -> pd.Series:
       """
       Fit m3: refit with original covariates if available in input data
       """
       #Pick predictor columns (if not specified by user)
       covariate_columns = self._resolve_covariate_columns(covariate_columns)

       if not covariate_columns:
           raise ValueError("No covariate columns provided to fit m3")
       covariate_formula = "+ ".join(covariate_columns)
       formula = f" {self.outcome_col} ~ {covariate_formula}"

       #Fit a new logistic regression model using all predictors,
       #re-estimating the intercept and every predictor's coefficient
       model = smf.glm(
           formula = formula,
           data = self.data,
           family = sm.families.Binomial(),
       ). fit(disp=0)

      #save fitted model
       self._fitted_m3 = model
      #store the predicted probabilities from the refitted model
       self.data["m3_pred"] = model.fittedvalues
       return self.data["m3_pred"]

   def get_predictions(self, covariate_columns: Optional[List[str]] = None) -> pd.DataFrame:
       """
       Get predictions from all models (m0-m3)
       """
       self.predict_m0()
       if "m1_pred" not in self.data.columns:
           self.fit_m1_recalibration_in_the_large()
       if "m2_pred" not in self.data.columns:
           self.fit_m2_logistic_calibration()
       if "m3_pred" not in self.data.columns:
           self.fit_m3_model_revision(covariate_columns=covariate_columns)
       return self.data[["m0_pred", "m1_pred", "m2_pred", "m3_pred"]].copy()

   @staticmethod
   def _log_likelihood(log_odds:np.ndarray, y: np.ndarray) -> float:
       """
       compute log-likelihood
       Bernoulli log-likelihood written in terms of log-odds:
       for each patient: y*lp - log(1+ e^lp), sum them all up
       """
       return np.sum(y*log_odds - np.log1p(np.exp(log_odds)))

   def compute_loglikelihoods(self) -> Dict[str, float]:
       """
       compute log-likelihoods for each model (m0-m3)
       """
       #Keep rows where only outcome and predictions are present
       data_refine = self.data.dropna(subset=[self.outcome_col, "predictions"])

       if len(data_refine) == 0:
           raise ValueError("No data to compute log-likelihoods")
       #observed outcomes
       y_true = data_refine[self.outcome_col].astype(float).values
       #m0
       #Keep predicted probabilities between 0 and 1
       predictions_m0 = np.clip(data_refine["predictions"].astype(float).values, 1e-15, 1 - 1e-15)
       #Convert probabilities to log-odds
       lp0 = np.log(predictions_m0 / (1 - predictions_m0))
       #log-likelihood of original model
       ll0 = self._log_likelihood(lp0, y_true)

       #m1
       #If not fitted, fit m1 first
       if self._fitted_m1 is None:
           self.fit_m1_recalibration_in_the_large()
       #Get estimated corrected intercept
       intercept_m1 = self._fitted_m1.params["Intercept"]
       #new log-odds (lp1)  = corrected + original log-odds (lp0)
       lp1 = intercept_m1 + self.data["linear_predictor"].values
       #log-likelihood of m1
       ll1 = self._log_likelihood(lp1, y_true)

       #m2
       if self._fitted_m2 is None:
           self.fit_m2_logistic_calibration()
       #predicted probabilities from m2
       prob2 = self._fitted_m2.fittedvalues.values
       prob2 = np.clip(prob2, 1e-15, 1 - 1e-15)
       lp2 = np.log(prob2 / (1-prob2))
       #log-likelihood of m2
       ll2 = self._log_likelihood(lp2, y_true)

       #m3
       if self._fitted_m3 is None:
           self.fit_m3_model_revision()
       #predicted probabilities form m3
       prob3 = self._fitted_m3.fittedvalues.values
       prob3 = np.clip(prob3, 1e-15, 1 - 1e-15)
       lp3 = np.log(prob3 / (1-prob3))
       #log-likelihood of m3
       ll3 = self._log_likelihood(lp3, y_true)
       return {"ll0": ll0, "ll1": ll1, "ll2": ll2, "ll3": ll3}

   def compute_closed_test_selection(self)-> Tuple[str,Dict[str,Any]]:

    lls = self.compute_loglikelihoods()
    ll0, ll1, ll2, ll3 = lls["ll0"], lls["ll1"], lls["ll2"], lls["ll3"]
    p = len(self._fitted_m3.params) - 1

    #calculate deviances compared with m3
    dev_original = -2* ll0 + 2 * ll3
    dev_intercept = -2* ll1 + 2* ll3
    dev_recalibration = -2* ll2 + 2* ll3

    #conduct tests at 0.05 significance level
    test_1 = chi2.sf(dev_original, df = p+1) < 0.05
    test_2 =chi2.sf(dev_intercept,df=p) < 0.05
    test_3 =chi2.sf(dev_recalibration,df=p-1) < 0.05

    #Model selection (closed-testing procedure)
    if not test_1:
        select = "m0"
    elif test_1 and not test_2:
        select = "m1"
    elif test_1 and test_2 and not test_3:
       select = "m2"
    else:
      select = "m3"

    stats = {
       "ll0": float(ll0),
       "ll1": float(ll1),
       "ll2": float(ll2),
       "ll3": float(ll3),
       "dev_original": dev_original,
       "dev_intercept": dev_intercept,
       "dev_recalibration": dev_recalibration,
       "test1_p_value":float(chi2.sf(dev_original,df=p+1)),
       "test2_p_value": float(chi2.sf(dev_intercept, df=p)),
       "test3_p_value": float(chi2.sf(dev_recalibration, df=p-1)),
       "test1_result": bool(test_1),
       "test2_result":bool(test_2),
       "test3_result":bool(test_3),
        "selected_model":select
}
    return select, stats

   def compute_roc_auc(self) -> Dict[str,Dict[str,Any]]:
     """
     Compute ROC AUC for m0-m3
     """

     self.get_predictions()
     #Keep rows where outcome is present
     data_refine = self.data.dropna(subset=[self.outcome_col])
     if len(data_refine)==0:
         raise ValueError("No data to compute ROC AUC for")
     #Observed outcomes
     y_true = data_refine[self.outcome_col].astype(float).values
     results = {}
     for model_name in [ "m0", "m1", "m2", "m3"]:
         #models predicted probabilities
         y_pred = data_refine[f"{model_name}_pred"].values
         #ROC curve computation
         fpr,tpr,thresholds = roc_curve(y_true, y_pred)
         #Area under ROC curve
         auc_val = roc_auc_score(y_true, y_pred)
         #Store results
         results[model_name] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": float(auc_val)}
     return results


   def get_model_coefficients(self, covariate_columns: Optional[List[str]] = None) -> Dict[str, Any]:
       """
       Return the fitted coefficients for each recalibration method.
       """
       self.drop_missing_predictions()
       self.get_predictions(covariate_columns=covariate_columns)
       covariate_columns = self._resolve_covariate_columns(covariate_columns)

       def get_coefficient(model, name):
           if model is not None and name in model.params.index:
               return float(model.params[name])
           return None

       return {
           "m1": {
               "intercept": get_coefficient(self._fitted_m1, "Intercept"),
           },
           "m2": {
               "intercept": get_coefficient(self._fitted_m2, "Intercept"),
               "slope": get_coefficient(self._fitted_m2, "linear_predictor"),
           },
           "m3": {
               "intercept": get_coefficient(self._fitted_m3, "Intercept"),
               "covariates": {
                   column: get_coefficient(self._fitted_m3, column)
                   for column in covariate_columns
               },
           },
       }

   def get_calibration_summary(self, covariate_columns: Optional[List[str]] = None) -> Dict[str,Any]:
       """
       Runs the full analysis (log-likelihoods, model selection, AUC), returning results in a dictionary
       """
       self.drop_missing_predictions()
       predictions = self.get_predictions(covariate_columns = covariate_columns)
       loglikelihoods = self.compute_loglikelihoods()
       selected_model, selection_stats = self.compute_closed_test_selection()
       roc_metrics = self.compute_roc_auc()
       calibration_curve_data = self.compute_calibration_curve()
       coefficients = self.get_model_coefficients(covariate_columns=covariate_columns)

       return {
           "predictions": predictions.to_dict(orient="list"),
           "log_likelihoods": loglikelihoods,
           "model_selection": {
           "selected_model": selected_model,
           "statistics": selection_stats
       },
           "roc_metrics": {
               name: {key: (val.tolist() if isinstance(val, np.ndarray) else val) for key, val in metrics.items()}
               for name, metrics in roc_metrics.items()
           },
           "calibration_curve": {
               name: {key: (val.tolist() if isinstance(val, np.ndarray) else val) for key, val in metrics.items()}
               for name, metrics in calibration_curve_data.items()
           },
           "coefficients": coefficients,
       }


   def get_results_table(self, covariate_columns: Optional[List[str]] = None) -> pd.DataFrame:
       """
        Builds a table of the fitted coefficients for each method
       """
       self.drop_missing_predictions()
       self.get_predictions(covariate_columns=covariate_columns)
       roc_metrics = self.compute_roc_auc()
       _, selection_stats = self.compute_closed_test_selection()
       rows = []
       covariate_columns = self._resolve_covariate_columns(covariate_columns)
       for var in covariate_columns:
           method3_val = (
               self._fitted_m3.params[var]
               if self._fitted_m3 is not None and var in self._fitted_m3.params.index
               else np.nan
           )
           rows.append({
               "Variable": var,
               "Original": "",
               "Method 1": "",
               "Method 2": "",
               "Method 3": round(method3_val, 3) if pd.notna(method3_val) else "",
           })

       m1_intercept = (
           self._fitted_m1.params["Intercept"]
           if self._fitted_m1 is not None and "Intercept" in self._fitted_m1.params.index
           else np.nan
       )
       m2_intercept = (
           self._fitted_m2.params["Intercept"]
           if self._fitted_m2 is not None and "Intercept" in self._fitted_m2.params.index
           else np.nan
       )

       m3_intercept = (
           self._fitted_m3.params["Intercept"]
           if self._fitted_m3 is not None and "Intercept" in self._fitted_m3.params.index
           else np.nan
       )

       rows.append(
           {
               "Variable": "Intercept",
               "Original": "",
               "Method 1": f"{round(m1_intercept, 4)}" if pd.notna(m1_intercept) else "",
               "Method 2": round(m2_intercept, 4) if pd.notna(m2_intercept) else "",
               "Method 3": round(m3_intercept, 4) if pd.notna(m3_intercept) else "",
           }
       )

       m2_slope = (
           self._fitted_m2.params["linear_predictor"]
           if self._fitted_m2 is not None and "linear_predictor" in self._fitted_m2.params.index
           else np.nan
       )

       rows.append(
           {
               "Variable": "Slope (linear_predictor)",
               "Original": "",
               "Method 1": "1",
               "Method 2": f"{round(m2_slope, 4)}" if pd.notna(m2_slope) else "",
               "Method 3": "",
           })

       rows.append(
           {
               "Variable": "AUC",
               "Original": round(roc_metrics["m0"]["auc"], 4) if "m0" in roc_metrics and "auc" in roc_metrics[
                   "m0"] else "",
               "Method 1": round(roc_metrics["m1"]["auc"], 4) if "m1" in roc_metrics and "auc" in roc_metrics[
                   "m1"] else "",
               "Method 2": round(roc_metrics["m2"]["auc"], 4) if "m2" in roc_metrics and "auc" in roc_metrics[
                   "m2"] else "",
               "Method 3": round(roc_metrics["m3"]["auc"], 4) if "m3" in roc_metrics and "auc" in roc_metrics[
                   "m3"] else "",

           }
       )

       return pd.DataFrame(rows)

   def compute_calibration_curve(self, n_bins: int = 10, strategy: str = "quantile"):
       """
       Group patients into bins and compare average predicted risk
       with observed outcome rate for each model
       """
       self.get_predictions()
       clean_data = self.data.dropna(subset=[self.outcome_col])
       y_true = clean_data[self.outcome_col].astype(float)
       results = {}
       for model in ["m0", "m1", "m2", "m3"]:
           prob_true, prob_pred = calibration_curve(y_true, clean_data[f"{model}_pred"], n_bins=n_bins,
                                                    strategy=strategy)
           results[model] = {"observed_probability": prob_true, "predicted_probability": prob_pred}

       return results

   def plot_calibration_curve(self, n_bins=5, strategy="quantile"):
       """
       Plot calibration curves for m0-m3, with histogram of predicted probabilities
       """

       curves = self.compute_calibration_curve(n_bins=n_bins, strategy=strategy)

       colors = {"m0": "black", "m1": "red", "m2": "green", "m3": "blue"}
       fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

       for model in curves:
           ax1.plot(
               curves[model]["predicted_probability"],
               curves[model]["observed_probability"],
               marker="o",
               color=colors[model],
               label=model.upper()
           )

       ax1.plot([0, 1], [0, 1], "--", color="gray")
       # ax1.xlabel("Predicted Probability")
       ax1.set_ylabel("Observed Probability")
       ax1.set_title("Calibration Curve Model")
       ax1.legend()
       ax1.set_ylim(0, 0.3)
       ax1.set_xlim(0, 0.3)
       ax1.grid(True)
       # Histogram of predicted probabilities
       self.get_predictions()
       for model in ["m0", "m1", "m2", "m3"]:
           ax2.hist(
               self.data[f"{model}_pred"].dropna(),
               bins=20,
               histtype="step",
               linewidth=2,
               color=colors[model],
               label=model.upper()
           )
       ax2.set_xlabel("Predicted Probability")
       ax2.set_xlim(0, 0.3)
       ax2.set_ylabel("Count")
       ax2.grid(True)
       plt.tight_layout()

       return fig

   def compute_classification_metrics(self, threshold=0.1):

       self.get_predictions()
       y_true = self.data[self.outcome_col].astype(int)

       results = {}

       for model in ["m0", "m1", "m2", "m3"]:
           y_prob = self.data[f"{model}_pred"]
           y_pred = (y_prob >= threshold).astype(int)

           tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

           sensitivity = tp / (tp + fn) if (tp+fn)> 0 else 0
           specificity = tn / (tn + fp) if (tn+fp)> 0 else 0
           precision = tp / (tp + fp) if (tp + fp)> 0 else 0
           recall = sensitivity
           f1 = 2 * precision * recall / (precision + recall) if (precision+recall)> 0 else 0
           accuracy = (tp + tn) / (tp + tn + fp + fn)

           results[model] = {

               "TP": tp,
               "TN": tn,
               "FP": fp,
               "FN": fn,

               "Sensitivity": sensitivity,
               "Specificity": specificity,
               "Precision": precision,
               "Recall": recall,
               "Accuracy": accuracy,
               "F1": f1
           }

       return results

   def compute_precision_recall(self):

       self.get_predictions()

       y_true = self.data[self.outcome_col]
       results = {}
       for model in ["m0", "m1", "m2", "m3"]:
           precision, recall, thresholds = precision_recall_curve(
               y_true,
               self.data[f"{model}_pred"]

           )

           results[model] = {
               "precision": precision,
               "recall": recall,
               "thresholds": thresholds
           }

       return results

   def plot_roc_curve(self):

       roc = self.compute_roc_auc()

       plt.figure(figsize=(7, 7))

       colors = {"m0": "black", "m1": "red", "m2": "green", "m3": "blue"}

       for model in roc:
           plt.plot(
               roc[model]["fpr"],
               roc[model]["tpr"],
               label=f"{model.upper()} (AUC = {roc[model]['auc']:.3f})",
               color=colors[model]

           )

       plt.plot([0, 1], [0, 1], "--")
       plt.xlabel("False Positive Rate")
       plt.ylabel("True Positive Rate")
       plt.title("ROC Curve Model")
       plt.legend()
       plt.grid(True)

       return plt.gcf()

   def plot_precision_recall_curve(self):

       pr = self.compute_precision_recall()
       plt.figure(figsize=(7, 7))
       colors = {"m0": "black", "m1": "red", "m2": "green", "m3": "blue"}
       for model in pr:
           plt.plot(pr[model]["recall"], pr[model]["precision"], color=colors[model], label=model.upper())

       plt.xlabel("Recall")
       plt.ylabel("Precision")
       plt.grid(True)
       plt.legend()
       plt.title("Precision-Recall Curve Model")
       return plt.gcf()

   def plot_all(self):

       self.plot_calibration_curve()
       self.plot_roc_curve()
       self.plot_precision_recall_curve()
       plt.show()

   def compute_metrics_all_thresholds(self, thresholds = None) -> Dict[str, pd.DataFrame]:
       """
       Compute classification metrics for all thresholds for all models m0-m3
       """
       self.get_predictions()
       if thresholds is None:
           thresholds = np.linspace(0.01, 0.99, 99)
       y_true = self.data[self.outcome_col].astype(int)
       results = {}
       for model in ["m0", "m1", "m2", "m3"]:
           y_prob = self.data[f"{model}_pred"]
           rows = []
           for threshold in thresholds:
               y_pred = (y_prob >= threshold).astype(int)
               tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
               sensitivity = tp / (tp + fn) if (tp+fn)> 0 else 0
               recall = sensitivity
               specificity = tn / (tn + fp) if (tn+fp)> 0 else 0
               precision = tp / (tp + fp) if (tp+fp)> 0 else 0
               f1 = 2 * precision * recall / (precision + recall) if (precision + recall)> 0 else 0
               accuracy = (tp + tn) / (tp + tn + fp + fn)
               rows.append({
                   "Threshold": threshold,
                   "Sensitivity": sensitivity,
                   "Specificity": specificity,
                   "Precision": precision,
                   "Recall": recall,
                   "F1": f1,
                   "Accuracy": accuracy
               })

           results[model] = pd.DataFrame(rows)
       return results

   def plot_metrics_all_thresholds(self, metric:str):
       """
       Plot one metric against threshold for all models
       """
       results = self.compute_metrics_all_thresholds()
       colors = {"m0": "black", "m1": "red", "m2": "green", "m3": "blue"}
       plt.figure(figsize=(7, 5))
       for model,df in results.items():
           plt.plot(df["Threshold"], df[metric], color=colors[model], label=model.upper())
       plt.xlabel("Threshold")
       plt.ylabel(metric)
       plt.title(f"{metric} vs Threshold Model")
       plt.legend()
       plt.grid(True)
       plt.tight_layout()
       return plt.gcf()

   def plot_threshold_analysis_all_models(self):
       self.plot_metrics_all_thresholds("Sensitivity")
       self.plot_metrics_all_thresholds("Specificity")
       self.plot_metrics_all_thresholds("Precision")
       self.plot_metrics_all_thresholds("F1")
       plt.show()

