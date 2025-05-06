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
    model_type = metadata_json.get("model_type", "classification")
    
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
    
    
@pytest.mark.parametrize("model_name", ["pilot-model_1"])#, "pilot-model_2"])
def test_metrics_calculator(shared_datadir, model_name):
    """
    Test the MetricsCalculator class with both pilot models and their corresponding CSV files.
    
    Parameters
    ----------
    shared_datadir : Path
        Pytest fixture providing access to shared test data directory
    model_name : str
        Name of the model to test (parameterized)
    """
    print(f"\n{'='*80}\nTESTING MODEL: {model_name}\n{'='*80}")
    
    model_dir = shared_datadir / "models"
    model_path = model_dir / model_name
    
    if not model_path.exists():
        print(f"! Model directory {model_name} not found, skipping test")
        pytest.skip(f"Model directory {model_name} not found")
    
    print(f"\n{'+'*40}")
    print(f"+ LOADING METADATA FOR MODEL: {model_name}")
    print(f"{'+'*40}")
    
    print(f"+ Loading model metadata from {model_path / 'metadata.json'}")
    metadata_json = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    
    model_type = metadata_json.get('model_type', 'classification')
    print(f"+ Model name: {model_metadata.model_name}")
    print(f"+ Model type: {model_type}")
    print(f"+ Docker image: {model_metadata.docker_image}")
    print(f"+ Output field: {model_metadata.output}")
    
    assert model_metadata.docker_image, "Docker image name should be provided"
    assert model_metadata.output, "Output field should be provided in model metadata"
    
    print(f"\n{'+'*40}")
    print(f"+ EXECUTING MODEL AND COLLECTING PREDICTIONS")
    print(f"{'+'*40}")
    
    csv_path = model_path / "data.csv"
    print(f"+ Loading input data from {csv_path}")
    inputs, expected_outputs = create_json_payloads(model_metadata, csv_path)
    print(f"+ Parsed {len(inputs)} input samples from CSV")
    
    # run model, fail if it doesn't work
    try:
        print(f"+ Executing model {model_name} Docker container...")
        predictions = execute_model(model_metadata, inputs)
        print(f"+ Successfully executed model and received {len(predictions)} predictions")
        print(f"+ First 3 predictions: {predictions[:3]}")
    except Exception as e:
        print(f"! Model execution failed for {model_name}: {e}")
        pytest.fail(f"Model execution failed for {model_name}: {e}")
    
    print(f"\n{'+'*40}")
    print(f"+ INITIALIZING METRICS CALCULATOR")
    print(f"{'+'*40}")
    
    calculator = MetricsCalculator(
        model_metadata=model_metadata,
        predictions=predictions,
        expected_outputs=expected_outputs,
        inputs=inputs
    )
    print(f"+ Created MetricsCalculator instance with {len(predictions)} prediction samples")
    
    column_metadata_path = model_path / "column_metadata.json"
    if not column_metadata_path.exists():
        print(f"! Column metadata file not found for {model_name}, skipping test")
        pytest.skip(f"Column metadata file not found for {model_name}")
    
    print(f"+ Loading categorical features from {column_metadata_path}")
    categorical_features = calculator.get_categorical_features_from_json(column_metadata_path)
    print(f"+ Found {len(categorical_features)} categorical features: {categorical_features}")
    
    print(f"\n{'+'*40}")
    print(f"+ PREPROCESSING AND VALIDATING PREDICTION DATA")
    print(f"{'+'*40}")
    
    y_true, y_pred, valid_indices = calculator.prepare_data()
    print(f"+ Total samples: {len(predictions)}, Valid numeric samples: {len(valid_indices)}")
    print(f"+ First 5 true values: {y_true[:5]}")
    print(f"+ First 5 predicted values: {y_pred[:5]}")
    print(f"+ Data types - y_true: {y_true.dtype}, y_pred: {y_pred.dtype}")
    
    assert len(y_true) > 0, f"Should have valid data points for {model_name}"
    assert len(y_true) == len(y_pred), "True values and predictions should have same length"
    
    print(f"\n{'+'*40}")
    print(f"+ ANALYZING MODEL OUTPUT FORMAT AND TYPE")
    print(f"{'+'*40}")
    
    is_prob_output = False
    if model_type.lower() == "classification":
        # check if predictions are valid prob values
        min_val = np.min(y_pred)
        max_val = np.max(y_pred)
        unique_vals = np.unique(y_pred)
        
        print(f"+ Model type: Classification")
        print(f"+ Value range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"+ Unique value examples: {unique_vals[:10]}")
        print(f"+ Number of unique values: {len(unique_vals)}")
        
        if np.all((y_pred >= 0) & (y_pred <= 1)):
            print(f"+ All values are in range [0, 1]")
            # check for the edge case where all values are exactly 0 or 1
            binary_vals = np.isin(y_pred, [0, 1])
            binary_percent = np.mean(binary_vals) * 100
            print(f"+ Percentage of exact 0 or 1 values: {binary_percent:.2f}%")
            
            if not np.all(binary_vals):
                is_prob_output = True
                print(f"+ OUTPUT TYPE: Probability values (containing values between 0-1)")
            else:
                print(f"+ OUTPUT TYPE: Binary classification (only 0 and 1 values)")
        else:
            print(f"+ OUTPUT TYPE: Non-standard classification values (outside [0,1] range)")
    else:
        print(f"+ Model type: Regression")
        print(f"+ Value range: [{np.min(y_pred):.6f}, {np.max(y_pred):.6f}]")
        print(f"+ Mean value: {np.mean(y_pred):.6f}")
        print(f"+ OUTPUT TYPE: Regression predictions (continuous values)")
    
    print(f"\n{'+'*40}")
    print(f"+ CALCULATING STANDARD PERFORMANCE METRICS")
    print(f"{'+'*40}")
    
    overall_metrics = calculator.calculate_metrics()
    assert overall_metrics, "Should return metrics dictionary"
    
    print(f"+ Generated {len(overall_metrics)} metrics:")
    for i, (key, value) in enumerate(overall_metrics.items()):
        print(f"  - {key}: {value}")
    
    # Test threshold metrics for probability outputs in classification
    if is_prob_output:
        print(f"\n{'+'*40}")
        print(f"+ CALCULATING THRESHOLD-BASED CLASSIFICATION METRICS")
        print(f"{'+'*40}")
        
        try:
            threshold_metrics = calculator.calculate_threshold_metrics()
            
            if "probability_preprocessing" in threshold_metrics:
                print(f"+ Probability preprocessing: {threshold_metrics['probability_preprocessing']}")
            
            print(f"+ ROC Curve Analysis:")
            if "roc_curve" in threshold_metrics:
                roc = threshold_metrics['roc_curve']
                print(f"  - AUC Score: {roc['auc']:.4f}")
                print(f"  - Thresholds analyzed: {len(roc['thresholds'])}")
                print(f"  - FPR range: [{min(roc['fpr']):.4f}, {max(roc['fpr']):.4f}]")
                print(f"  - TPR range: [{min(roc['tpr']):.4f}, {max(roc['tpr']):.4f}]")
            
            print(f"+ Precision-Recall Curve Analysis:")
            if "pr_curve" in threshold_metrics:
                pr = threshold_metrics['pr_curve']
                print(f"  - Average Precision: {pr['average_precision']:.4f}")
                print(f"  - Thresholds analyzed: {len(pr['thresholds'])}")
                print(f"  - Precision range: [{min(pr['precision']):.4f}, {max(pr['precision']):.4f}]")
                print(f"  - Recall range: [{min(pr['recall']):.4f}, {max(pr['recall']):.4f}]")
            
            print(f"+ Performance at Selected Decision Thresholds:")
            sample_thresholds = ["0.1", "0.3", "0.5", "0.7", "0.9"]
            for t in sample_thresholds:
                if t in threshold_metrics.get('threshold_metrics', {}):
                    tm = threshold_metrics['threshold_metrics'][t]
                    print(f"  - Threshold {t}:")
                    print(f"    * Accuracy: {tm['accuracy']:.4f}")
                    print(f"    * Precision: {tm['precision']:.4f}")
                    print(f"    * Recall: {tm['recall']:.4f}")
                    print(f"    * F1-score: {tm['f1_score']:.4f}")
                    print(f"    * Confusion Matrix: TP={tm['confusion_matrix']['tp']}, " +
                         f"TN={tm['confusion_matrix']['tn']}, " +
                         f"FP={tm['confusion_matrix']['fp']}, " +
                         f"FN={tm['confusion_matrix']['fn']}")
            
            assert "roc_curve" in threshold_metrics, "Should contain ROC curve data"
            assert "pr_curve" in threshold_metrics, "Should contain PR curve data"
            assert "threshold_metrics" in threshold_metrics, "Should contain threshold metrics"
        except Exception as e:
            print(f"! Failed to calculate threshold metrics: {e}")
            pytest.fail(f"Failed to calculate threshold metrics for {model_name}: {e}")
    
    print(f"\n{'+'*40}")
    print(f"+ CALCULATING COMPLETE METRICS SUITE (OVERALL & SUBGROUPS)")
    print(f"{'+'*40}")
    
    all_metrics = calculator.calculate_all_metrics_from_json(csv_path, column_metadata_path)
    
    print(f"+ Metrics result structure validation:")
    assert "model_info" in all_metrics, "Should contain model info"
    assert "overall" in all_metrics, "Should contain overall metrics"
    
    print(f"+ Model info:")
    for key, value in all_metrics["model_info"].items():
        print(f"  - {key}: {value}")
    
    print(f"+ Overall metrics contains {len(all_metrics['overall'])} metrics")
    
    # ff classification with probability outputs, check for threshold metrics
    if is_prob_output and "threshold_metrics" in all_metrics:
        print(f"+ Threshold metrics included: Yes")
        tm_keys = list(all_metrics["threshold_metrics"].keys())
        print(f"  - Sections: {tm_keys}")
    
    # check if subgroup analysis was performed for cat features
    if "subgroups" in all_metrics:
        print(f"\n{'+'*40}")
        print(f"+ EXAMINING SUBGROUP ANALYSIS RESULTS")
        print(f"{'+'*40}")
        
        subgroups = all_metrics["subgroups"]
        print(f"+ Subgroup analysis performed for {len(subgroups)} features: {list(subgroups.keys())}")
        
        for feature in categorical_features:
            if feature in subgroups:
                if not isinstance(subgroups[feature], dict) or "error" in subgroups.get(feature, {}):
                    print(f"! '{feature}' may not be a valid categorical column in the dataset: {subgroups.get(feature)}")
                else:
                    print(f"+ Feature '{feature}' has {len(subgroups[feature])} subgroups")
                    for value, metrics in subgroups.get(feature, {}).items():
                        print(f"  - Subgroup '{value}': sample_size={metrics.get('sample_size')}")
                        print(f"    * {len(metrics.keys())} metrics calculated")
            else:
                print(f"! Feature '{feature}' not found in subgroups")
    else:
        print(f"! No subgroups section found in metrics output")
    
    print(f"\n{'+'*40}")
    print(f"+ SAVING AND VALIDATING METRICS JSON OUTPUT")
    print(f"{'+'*40}")
    
    output_path = shared_datadir / f"test_metrics_output_{model_name}.json"
    print(f"+ Saving comprehensive metrics to {output_path}")
    saved_metrics = calculator.save_metrics_to_json_from_metadata(output_path, csv_path, column_metadata_path)
    
    assert output_path.exists(), f"JSON file should be created for {model_name}"
    print(f"+ JSON file successfully created at {output_path}")
    
    # verify saved file
    print(f"+ Validating saved JSON content:")
    loaded_metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded_metrics["model_info"]["name"] == model_metadata.model_name, "Saved JSON should contain correct model name"
    print(f"  - Model name: {loaded_metrics['model_info']['name']}")
    print(f"  - Model type: {loaded_metrics['model_info']['type']}")
    print(f"  - Contains overall metrics: {len(loaded_metrics['overall'])} metrics")
    if "threshold_metrics" in loaded_metrics:
        print(f"  - Contains threshold metrics: Yes")
    if "subgroups" in loaded_metrics:
        print(f"  - Contains subgroup analysis: Yes, for {len(loaded_metrics['subgroups'])} features")
    
    # clean stuff up
    output_path.unlink(missing_ok=True)    
    print(f"\n{'='*80}")
    print(f"TEST COMPLETED SUCCESSFULLY FOR MODEL: {model_name}")
    print(f"{'='*80}")