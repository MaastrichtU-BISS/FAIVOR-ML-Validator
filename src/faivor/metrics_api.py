import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple

from faivor.model_metadata import ModelMetadata
from faivor.parse_data import detect_delimiter
from faivor.metrics.regression import metrics as regression_metrics
from faivor.metrics.classification import metrics as classification_metrics


class MetricsCalculator:
    """
    Calculate metrics for model predictions and perform subgroup analysis.
    """
    
    def __init__(self, model_metadata: ModelMetadata, predictions: List, 
                expected_outputs: List, inputs: Optional[List] = None):
        """
        Initialize the metrics calculator.
        
        Parameters
        ----------
        model_metadata : ModelMetadata
            Model metadata containing information about the model.
        predictions : List
            Predictions from the model.
        expected_outputs : List
            Expected outputs (ground truth) for evaluation.
        inputs : List, optional
            Input data used for making predictions, by default None.
        """
        self.model_metadata = model_metadata
        self.predictions = predictions
        self.expected_outputs = expected_outputs
        self.inputs = inputs
        self.output_field = model_metadata.output
        self.metadata_json = model_metadata.metadata
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare data for metrics calculation by filtering valid samples.
        
        Returns
        -------
        tuple
            Tuple containing (y_true, y_pred, valid_indices)
        """
        valid_indices = []
        y_true_values = []
        y_pred_values = []
        
        for i, (output, pred) in enumerate(zip(self.expected_outputs, self.predictions)):
            try:
                # need to check if expected output has the required key and is numeric
                if isinstance(output, dict) and self.output_field in output:
                    true_val = output[self.output_field]
                    if true_val in ('x', 'X', ''):
                        continue
                    
                    true_float = float(true_val)
                    pred_float = float(pred)
                    
                    valid_indices.append(i)
                    y_true_values.append(true_float)
                    y_pred_values.append(pred_float)
                else:
                    continue
            except (ValueError, TypeError):
                continue
        
        y_true = np.array(y_true_values)
        y_pred = np.array(y_pred_values)
        
        return y_true, y_pred, valid_indices
    
    def get_sensitive_values(self, valid_indices: List[int]) -> Optional[np.ndarray]:
        """
        Get sensitive attribute values for valid indices if available.
        
        Parameters
        ----------
        valid_indices : List[int]
            Indices of valid data points.
            
        Returns
        -------
        np.ndarray or None
            Array of sensitive values if available, None otherwise.
        """
        sensitive_attribute = self.metadata_json.get("sensitive_attribute")
        if not sensitive_attribute or not self.inputs:
            return None
            
        sensitive_values = []
        for i in valid_indices:
            if isinstance(self.inputs[i], dict) and sensitive_attribute in self.inputs[i]:
                sensitive_values.append(self.inputs[i][sensitive_attribute])
            else:
                sensitive_values.append(None)
        
        if any(v is not None for v in sensitive_values):
            return np.array(sensitive_values)
        return None
    
    @staticmethod
    def _convert_to_json_serializable(obj: Any) -> Any:
        """
        Convert NumPy and other non-serializable objects to JSON serializable types.
        
        Parameters
        ----------
        obj : Any
            Object to convert
            
        Returns
        -------
        Any
            JSON serializable object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj
    
    def get_categorical_features_from_json(self, json_path: Path) -> List[str]:
        """
        Read a JSON file that describes column metadata and extract categorical column names.
        
        Parameters
        ----------
        json_path : Path
            Path to the JSON file with column metadata.
            
        Returns
        -------
        List[str]
            List of categorical column names.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            column_metadata = json.load(f)
        
        categorical_columns = []
        for column in column_metadata.get("columns", []):
            if column.get("categorical", False):
                categorical_columns.append(column.get("name"))
        
        return categorical_columns    
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics for the model.
        
        Returns
        -------
        dict
            Dictionary containing calculated metrics.
        """
        y_true, y_pred, valid_indices = self.prepare_data()
        
        if len(y_true) == 0:
            return {"error": "No valid numeric data pairs found for metric calculation"}
            
        model_type = self.metadata_json.get("model_type", "regression")
        
        # sensitive attribute - TODO:need to be mentiuoned in the schema --- from frontend, not fairmodels
        sensitive_values = self.get_sensitive_values(valid_indices)
        feature_importance = self.metadata_json.get("feature_importance")
        
        all_metrics = {}
        
        # detect appropriate metrics module based on model type
        metrics_module = (classification_metrics if model_type.lower() == "classification" 
                         else regression_metrics)
        
        for metric in metrics_module.performance:
            try:
                result = metric.compute(y_true, y_pred)
                all_metrics[f"performance.{metric.regular_name}"] = self._convert_to_json_serializable(result)
            except Exception as e:
                all_metrics[f"performance.{metric.regular_name}"] = f"Error: {str(e)}"
        
        for metric in metrics_module.fairness:
            try:
                if metric.function_name == "demographic_parity_ratio":
                    if sensitive_values is not None:
                        result = metric.compute(y_true, y_pred, sensitive_values)
                        all_metrics[f"fairness.{metric.regular_name}"] = self._convert_to_json_serializable(result)
                    else:
                        all_metrics[f"fairness.{metric.regular_name}"] = "Requires sensitive attributes (not computed)"
                    continue
                    
                result = metric.compute(y_true, y_pred)
                all_metrics[f"fairness.{metric.regular_name}"] = self._convert_to_json_serializable(result)
            except Exception as e:
                all_metrics[f"fairness.{metric.regular_name}"] = f"Error: {str(e)}"
        
        for metric in metrics_module.explainability:
            try:
                if metric.function_name == "feature_importance_ratio":
                    if feature_importance:
                        result = metric.compute(y_true, y_pred, feature_importance=feature_importance)
                        all_metrics[f"explainability.{metric.regular_name}"] = self._convert_to_json_serializable(result)
                    else:
                        all_metrics[f"explainability.{metric.regular_name}"] = "Requires feature importances (not computed)"
                    continue
                    
                result = metric.compute(y_true, y_pred)
                all_metrics[f"explainability.{metric.regular_name}"] = self._convert_to_json_serializable(result)
            except Exception as e:
                all_metrics[f"explainability.{metric.regular_name}"] = f"Error: {str(e)}"
                
        return all_metrics
        
    def calculate_subgroup_metrics(self, csv_path: Path, 
                                categorical_features: List[str]) -> Dict[str, Any]:
        """
        Calculate metrics for subgroups based on categorical features.
        
        Parameters
        ----------
        csv_path : Path
            Path to the CSV file containing the data.
        categorical_features : List[str]
            List of categorical feature names.
            
        Returns
        -------
        dict
            Dictionary containing metrics for each subgroup.
        """
        # handle delimiter and read the CSV file
        delimiter = detect_delimiter(csv_path)
        df = pd.read_csv(csv_path, sep=delimiter)
        
        y_true, y_pred, valid_indices = self.prepare_data()
        
        if len(y_true) == 0:
            return {"error": "No valid numeric data pairs found for subgroup metric calculation"}
        
        model_type = self.metadata_json.get("model_type", "regression")
        metrics_module = (classification_metrics if model_type.lower() == "classification" 
                        else regression_metrics)
        
        feature_importance = self.metadata_json.get("feature_importance")
        
        valid_data = df.iloc[valid_indices].reset_index(drop=True)
        
        valid_data['y_true'] = y_true
        valid_data['y_pred'] = y_pred
        
        # ff we have sensitive attribute info, add it to valid_data
        sensitive_attribute = self.metadata_json.get("sensitive_attribute")
        has_sensitive_data = False
        
        if sensitive_attribute and self.inputs:
            sensitive_values = []
            for i in valid_indices:
                if isinstance(self.inputs[i], dict) and sensitive_attribute in self.inputs[i]:
                    sensitive_values.append(self.inputs[i][sensitive_attribute])
                else:
                    sensitive_values.append(None)
            
            if any(v is not None for v in sensitive_values):
                valid_data['sensitive_attr'] = sensitive_values
                has_sensitive_data = True
        
        subgroup_metrics = {}
        
        for feature in categorical_features:
            if feature not in valid_data.columns:
                subgroup_metrics[feature] = {"error": f"Feature '{feature}' not found in data"}
                continue
                
            feature_values = valid_data[feature].unique()
            feature_metrics = {}
            
            for value in feature_values:
                # filter data for this subgroup - refactored from lambda function for readability
                subgroup_data = valid_data[valid_data[feature] == value]
                
                # important to skip if no data in this subgroup, e.g. empty or nan or invalid values
                if len(subgroup_data) == 0:
                    continue
                    
                subgroup_y_true = subgroup_data['y_true'].values
                subgroup_y_pred = subgroup_data['y_pred'].values
                
                subgroup_sensitive = subgroup_data['sensitive_attr'].values if has_sensitive_data else None
                
                subgroup_all_metrics = {
                    "sample_size": len(subgroup_y_true)  # Include sample size for context
                }
                
                for metric in metrics_module.performance:
                    try:
                        result = metric.compute(subgroup_y_true, subgroup_y_pred)
                        subgroup_all_metrics[f"performance.{metric.regular_name}"] = self._convert_to_json_serializable(result)
                    except Exception as e:
                        subgroup_all_metrics[f"performance.{metric.regular_name}"] = f"Error: {str(e)}"
                
                for metric in metrics_module.fairness:
                    try:
                        if metric.function_name == "demographic_parity_ratio":
                            if subgroup_sensitive is not None:
                                result = metric.compute(subgroup_y_true, subgroup_y_pred, subgroup_sensitive)
                                subgroup_all_metrics[f"fairness.{metric.regular_name}"] = self._convert_to_json_serializable(result)
                            else:
                                subgroup_all_metrics[f"fairness.{metric.regular_name}"] = "Requires sensitive attributes (not computed)"
                            continue
                            
                        result = metric.compute(subgroup_y_true, subgroup_y_pred)
                        subgroup_all_metrics[f"fairness.{metric.regular_name}"] = self._convert_to_json_serializable(result)
                    except Exception as e:
                        subgroup_all_metrics[f"fairness.{metric.regular_name}"] = f"Error: {str(e)}"
                
                for metric in metrics_module.explainability:
                    try:
                        if metric.function_name == "feature_importance_ratio":
                            if feature_importance:
                                result = metric.compute(subgroup_y_true, subgroup_y_pred, feature_importance=feature_importance)
                                subgroup_all_metrics[f"explainability.{metric.regular_name}"] = self._convert_to_json_serializable(result)
                            else:
                                subgroup_all_metrics[f"explainability.{metric.regular_name}"] = "Requires feature importances (not computed)"
                            continue
                            
                        result = metric.compute(subgroup_y_true, subgroup_y_pred)
                        subgroup_all_metrics[f"explainability.{metric.regular_name}"] = self._convert_to_json_serializable(result)
                    except Exception as e:
                        subgroup_all_metrics[f"explainability.{metric.regular_name}"] = f"Error: {str(e)}"
                
                feature_metrics[str(value)] = subgroup_all_metrics
                
            subgroup_metrics[feature] = feature_metrics
            
        return subgroup_metrics
    
    def calculate_all_metrics(self, csv_path: Optional[Path] = None, 
                             categorical_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate all metrics including overall and subgroup metrics.
        
        Parameters
        ----------
        csv_path : Path, optional
            Path to the CSV file, by default None
        categorical_features : List[str], optional
            List of categorical feature names, by default None
            
        Returns
        -------
        dict
            Dictionary containing all metrics.
        """
        result = {
            "model_info": {
                "name": self.model_metadata.model_name,
                "type": self.metadata_json.get("model_type", "regression")
            },
            "overall": self.calculate_metrics()
        }
        
        if csv_path and categorical_features:
            result["subgroups"] = self.calculate_subgroup_metrics(csv_path, categorical_features)
            
        return result
    
    def save_metrics_to_json(self, output_path: Path, csv_path: Optional[Path] = None, 
                            categorical_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate all metrics and save them to a JSON file.
        
        Parameters
        ----------
        output_path : Path
            Path to save the JSON file.
        csv_path : Path, optional
            Path to the CSV file, by default None
        categorical_features : List[str], optional
            List of categorical feature names, by default None
            
        Returns
        -------
        dict
            Dictionary containing all metrics.
        """
        metrics = self.calculate_all_metrics(csv_path, categorical_features)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
            
        return metrics