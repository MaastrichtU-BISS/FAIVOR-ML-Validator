import json
import numpy as np
import pytest
from faivor.model_metadata import ModelMetadata
from faivor.parse_data import create_json_payloads
from faivor.run_docker import execute_model
from faivor.model_metadata import ModelMetadata
from faivor.parse_data import create_json_payloads
from faivor.run_docker import execute_model
from faivor.metrics.regression import metrics as regression_metrics
from faivor.metrics.classification import metrics as classification_metrics
import logging
from faivor.metrics_api import MetricsCalculator
from pathlib import Path
from typing import Dict, Any

model:str = "pilot-model_1"

def test_model_execution(shared_datadir):
    model_dir = shared_datadir / "models"
    metadata_json = json.loads((model_dir / model / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    assert model_metadata.docker_image, "Docker image name should be provided"
    csv_path = model_dir / model / "data.csv"
    inputs, _ = create_json_payloads(model_metadata, csv_path)
    try:
        prediction = execute_model(model_metadata, inputs)
    except Exception as e:
        raise RuntimeError(f"Model execution failed: {e}")


    assert prediction is not None, "Model execution should return a prediction."
    assert isinstance(prediction, list), "Prediction result should be a dictionary."



def test_model_metrics(shared_datadir):
    
    model_dir = shared_datadir / "models"
    metadata_json = json.loads((model_dir / model / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    assert model_metadata.docker_image, "Docker image name should be provided"

    output_field = model_metadata.output
    assert output_field, "Output field should be provided in model metadata"
    
    csv_path = model_dir / model / "data.csv"
    inputs, expected_outputs = create_json_payloads(model_metadata, csv_path)
    
    try:
        predictions = execute_model(model_metadata, inputs)
    except Exception as e:
        raise RuntimeError(f"Model execution failed: {e}")

    valid_indices = []
    y_true_values = []
    y_pred_values = []
    
    for i, (output, pred) in enumerate(zip(expected_outputs, predictions)):
        try:
            # expected output has the required key and is numeric?
            if isinstance(output, dict) and output_field in output:
                true_val = output[output_field]
                if true_val == 'x' or true_val == 'X' or true_val == '':
                    print(f"Skipping non-numeric expected output at index {i}: {true_val}")
                    continue
                
                true_float = float(true_val)
                pred_float = float(pred)
                
                valid_indices.append(i)
                y_true_values.append(true_float)
                y_pred_values.append(pred_float)
            else:
                print(f"Expected output at index {i} missing '{output_field}' key: {output}")
        except (ValueError, TypeError) as e:
            print(f"Skipping this sample. Error at index {i}: {e}")
            print(f"Expected output: {output}, Prediction: {pred}")
 
    
    y_true = np.array(y_true_values)
    y_pred = np.array(y_pred_values)
    
    print(f"Total samples: {len(expected_outputs)}, Valid samples after filtering: {len(valid_indices)}")
    
    # enough valid data for metrics
    if len(y_true) == 0:
        assert False, "No valid numeric data pairs found for metric calculation"
        
    assert len(y_true) == len(y_pred), "Number of predictions doesn't match number of expected outputs"
    
    # check if model_type is provided in metadata, otherwise default to regression
    model_type = metadata_json.get("model_type", "regression")
    
    # check if sensitive_attribute is provided in metadata
    sensitive_attribute = metadata_json.get("sensitive_attribute")
    if sensitive_attribute:
        print(f"Found sensitive attribute in metadata: {sensitive_attribute}")
        # check if we have sensitive attribute values in the input data
        sensitive_values = []
        for i in valid_indices:
            if isinstance(inputs[i], dict) and sensitive_attribute in inputs[i]:
                sensitive_values.append(inputs[i][sensitive_attribute])
            else:
                sensitive_values.append(None)
        
        if any(v is not None for v in sensitive_values):
            sensitive_values = np.array(sensitive_values)
            print(f"Using sensitive attribute values: {sensitive_values}")
        else:
            sensitive_values = None
            print("Sensitive attribute not found in input data")
    else:
        sensitive_values = None
        print("No sensitive attribute defined in metadata")
    
    # check if feature_importance is provided in metadata
    feature_importance = metadata_json.get("feature_importance")
    if feature_importance:
        print(f"Found feature importance in metadata: {feature_importance}")
    
    print(f"\nModel type detected: {model_type}")
    
    all_metrics = {}
    
    if model_type.lower() == "classification":
                
        print("----- Classification Performance Metrics -----")
        for metric in classification_metrics.performance:
            try:
                result = metric.compute(y_true, y_pred)
                all_metrics[f"performance.{metric.regular_name}"] = result
                print(f"+ Performance metric - {metric.regular_name}: {result}")
            except Exception as e:
                print(f"Could not compute {metric.regular_name}: {e}")
        
        print("----- Classification Fairness Metrics -----")
        for metric in classification_metrics.fairness:
            try:
                if metric.function_name == "demographic_parity_ratio":
                    # check if we have sensitive attribute data
                    if sensitive_values is not None:
                        result = metric.compute(y_true, y_pred, sensitive_values)
                        all_metrics[f"fairness.{metric.regular_name}"] = result
                        print(f"+ Fairness metric - {metric.regular_name}: {result}")
                    else:
                        print(f"+ Fairness metric - {metric.regular_name}: requires sensitive attributes (not computed)")
                    continue
                    
                result = metric.compute(y_true, y_pred)
                all_metrics[f"fairness.{metric.regular_name}"] = result
                print(f"+ Fairness metric - {metric.regular_name}: {result}")
            except Exception as e:
                print(f"Could not compute {metric.regular_name}: {e}")
        
        print("----- Classification Explainability Metrics -----")
        for metric in classification_metrics.explainability:
            try:
                if metric.function_name == "feature_importance_ratio":
                    # check if we have feature importance data
                    if feature_importance:
                        result = metric.compute(y_true, y_pred, feature_importance=feature_importance)
                        all_metrics[f"explainability.{metric.regular_name}"] = result
                        print(f"+ Explainability metric - {metric.regular_name}: {result}")
                    else:
                        print(f"+ Explainability metric - {metric.regular_name}: requires feature importances (not computed)")
                    continue
                    
                result = metric.compute(y_true, y_pred)
                all_metrics[f"explainability.{metric.regular_name}"] = result
                print(f"+ Explainability metric - {metric.regular_name}: {result}")
            except Exception as e:
                print(f"\nCould not compute {metric.regular_name}: {e}")
                
    else:  # Regression metrics

        print("----- Regression Performance Metrics -----")
        for metric in regression_metrics.performance:
            try:
                result = metric.compute(y_true, y_pred)
                all_metrics[f"performance.{metric.regular_name}"] = result
                print(f"+ Performance metric - {metric.regular_name}: {result}")
            except Exception as e:
                print(f"Could not compute {metric.regular_name}: {e}")
        
        print("----- Regression Fairness Metrics -----")
        for metric in regression_metrics.fairness:
            try:
                if metric.function_name == "demographic_parity_ratio":
                    # check if we have sensitive attribute data
                    if sensitive_values is not None:
                        result = metric.compute(y_true, y_pred, sensitive_values)
                        all_metrics[f"fairness.{metric.regular_name}"] = result
                        print(f"+ Fairness metric - {metric.regular_name}: {result}")
                    else:
                        print(f"+ Fairness metric - {metric.regular_name}: requires sensitive attributes (not computed)")
                    continue
                    
                result = metric.compute(y_true, y_pred)
                all_metrics[f"fairness.{metric.regular_name}"] = result
                print(f"+ Fairness metric - {metric.regular_name}: {result}")
            except Exception as e:
                print(f"Could not compute {metric.regular_name}: {e}")
        
        print("----- Regression Explainability Metrics -----")
        for metric in regression_metrics.explainability:
            try:
                if metric.function_name == "feature_importance_ratio":
                    # check if we have feature importance data
                    if feature_importance:
                        result = metric.compute(y_true, y_pred, feature_importance=feature_importance)
                        all_metrics[f"explainability.{metric.regular_name}"] = result
                        print(f"+ Explainability metric - {metric.regular_name}: {result}")
                    else:
                        print(f"+ Explainability metric - {metric.regular_name}: requires feature importances (not computed)")
                    continue
                    
                result = metric.compute(y_true, y_pred)
                all_metrics[f"explainability.{metric.regular_name}"] = result
                print(f"+ Explainability metric - {metric.regular_name}: {result}")
            except Exception as e:
                print(f"Could not compute {metric.regular_name}: {e}")
    
    print(f"\nSuccessfully calculated {len(all_metrics)} metrics")
    
    assert True
    
    
def test_metrics_calculator(shared_datadir):
    """
    Test the MetricsCalculator class with a model and its corresponding CSV.
    """
    model = "pilot-model_1"
    model_dir = shared_datadir / "models"
    model_path = model_dir / model
    
    metadata_json = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    
    # validate metadata
    assert model_metadata.docker_image, "Docker image name should be provided"
    assert model_metadata.output, "Output field should be provided in model metadata"
    
    csv_path = model_path / "data.csv"
    inputs, expected_outputs = create_json_payloads(model_metadata, csv_path)
    
    # run model, get predictions
    try:
        predictions = execute_model(model_metadata, inputs)
    except Exception as e:
        pytest.fail(f"Model execution failed: {e}")
    
    calculator = MetricsCalculator(
        model_metadata=model_metadata,
        predictions=predictions,
        expected_outputs=expected_outputs,
        inputs=inputs
    )
    
    # Get categorical features from column metadata
    column_metadata_path = model_path / "column_metadata.json"
    categorical_features = calculator.get_categorical_features_from_json(column_metadata_path)
    
    # test the prepare_data method
    y_true, y_pred, valid_indices = calculator.prepare_data()
    assert len(y_true) > 0, "Should have valid data points"
    assert len(y_true) == len(y_pred), "True values and predictions should have same length"
    
    # test the calculate_metrics method
    overall_metrics = calculator.calculate_metrics()
    assert overall_metrics, "Should return metrics dictionary"
    assert "performance" in str(overall_metrics), "Should contain performance metrics"
    
    # test with categorical features from JSON metadata
    all_metrics = calculator.calculate_all_metrics(csv_path, categorical_features)
    
    # check the structure
    assert "model_info" in all_metrics, "Should contain model info"
    assert "overall" in all_metrics, "Should contain overall metrics"
    
    # check if subgroup analysis was performed for categorical features
    if "subgroups" in all_metrics:
        subgroups = all_metrics["subgroups"]
        for feature in categorical_features:
            if feature in subgroups and not isinstance(subgroups[feature], dict) or "error" in subgroups.get(feature, {}):
                print(f"'{feature}' may not be a valid categorical column in the dataset: {subgroups.get(feature)}")
            else:
                assert isinstance(subgroups.get(feature, {}), dict), f"Should contain {feature} subgroups"
                for value, metrics in subgroups.get(feature, {}).items():
                    assert "sample_size" in metrics, f"Each subgroup should report sample size (group '{value}')"
                    assert any("performance" in key for key in metrics.keys()), f"Each subgroup should have performance metrics (group '{value}')"
    
    # Testing the JSON file output
    output_path = shared_datadir / "test_metrics_output.json"
    saved_metrics = calculator.save_metrics_to_json(output_path, csv_path, categorical_features)
    
    assert output_path.exists(), "JSON file should be created"
    
    # verify saved file
    loaded_metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded_metrics["model_info"]["name"] == model_metadata.model_name, "Saved JSON should contain correct model name"
    
    # clean up
    output_path.unlink(missing_ok=True)
    

def calculate_all_metrics_from_json(self, csv_path: Path, column_metadata_path: Path) -> Dict[str, Any]:
    """
    Calculate all metrics including overall and subgroup metrics using column metadata.
    
    Parameters
    ----------
    csv_path : Path
        Path to the CSV file
    column_metadata_path : Path
        Path to the column metadata JSON file
            
    Returns
    -------
    dict
        Dictionary containing all metrics.
    """
    categorical_features = self.get_categorical_features_from_json(column_metadata_path)
    return self.calculate_all_metrics(csv_path, categorical_features)

def save_metrics_to_json_from_metadata(self, output_path: Path, csv_path: Path, 
                                     column_metadata_path: Path) -> Dict[str, Any]:
    """
    Calculate all metrics and save them to a JSON file using column metadata.
    
    Parameters
    ----------
    output_path : Path
        Path to save the JSON file.
    csv_path : Path
        Path to the CSV file
    column_metadata_path : Path
        Path to the column metadata JSON file
            
    Returns
    -------
    dict
        Dictionary containing all metrics.
    """
    categorical_features = self.get_categorical_features_from_json(column_metadata_path)
    return self.save_metrics_to_json(output_path, csv_path, categorical_features)