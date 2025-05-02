import json
import numpy as np
import pytest
from pathlib import Path

from faivor.model_metadata import ModelMetadata
from faivor.parse_data import create_json_payloads
from faivor.run_docker import execute_model
from faivor.metrics_api import MetricsCalculator


def test_metrics_calculator(shared_datadir):
    """
    Test the MetricsCalculator class with a model and its corresponding CSV.
    """
    # Setup - use the same model as in test_model_execution.py
    model = "pilot-model_1"
    model_dir = shared_datadir / "models"
    model_path = model_dir / model
    
    # Load model metadata
    metadata_json = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    
    # Validate model metadata
    assert model_metadata.docker_image, "Docker image name should be provided"
    assert model_metadata.output, "Output field should be provided in model metadata"
    
    # Load CSV data and create input/output pairs
    csv_path = model_path / "data.csv"
    inputs, expected_outputs = create_json_payloads(model_metadata, csv_path)
    
    # Execute model to get predictions
    try:
        predictions = execute_model(model_metadata, inputs)
    except Exception as e:
        pytest.fail(f"Model execution failed: {e}")
    
    # Create the MetricsCalculator instance
    calculator = MetricsCalculator(
        model_metadata=model_metadata,
        predictions=predictions,
        expected_outputs=expected_outputs,
        inputs=inputs
    )
    
    # Test the prepare_data method
    y_true, y_pred, valid_indices = calculator.prepare_data()
    assert len(y_true) > 0, "Should have valid data points"
    assert len(y_true) == len(y_pred), "True values and predictions should have same length"
    
    # Test the calculate_metrics method
    overall_metrics = calculator.calculate_metrics()
    assert overall_metrics, "Should return metrics dictionary"
    assert "performance" in str(overall_metrics), "Should contain performance metrics"
    
    # Test with categorical features - assuming 'sex' is a categorical column
    categorical_features = ["sex"]
    all_metrics = calculator.calculate_all_metrics(csv_path, categorical_features)
    
    # Check the structure of the returned metrics
    assert "model_info" in all_metrics, "Should contain model info"
    assert "overall" in all_metrics, "Should contain overall metrics"
    
    # Check if subgroup analysis was performed (if 'sex' column exists)
    if "subgroups" in all_metrics:
        subgroups = all_metrics["subgroups"]
        if "sex" in subgroups and not isinstance(subgroups["sex"], dict) or "error" in subgroups.get("sex", {}):
            print(f"Note: 'sex' may not be a valid categorical column in the dataset: {subgroups.get('sex')}")
        else:
            assert isinstance(subgroups.get("sex", {}), dict), "Should contain sex subgroups"
            for value, metrics in subgroups.get("sex", {}).items():
                assert "sample_size" in metrics, f"Each subgroup should report sample size (group '{value}')"
                assert any("performance" in key for key in metrics.keys()), f"Each subgroup should have performance metrics (group '{value}')"
    
    # Test saving to JSON
    output_path = shared_datadir / "test_metrics_output.json"
    saved_metrics = calculator.save_metrics_to_json(output_path, csv_path, categorical_features)
    
    # Verify the file was created
    assert output_path.exists(), "JSON file should be created"
    
    # Load and verify the saved file
    loaded_metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded_metrics["model_info"]["name"] == model_metadata.model_name, "Saved JSON should contain correct model name"
    
    # Clean up
    output_path.unlink(missing_ok=True)