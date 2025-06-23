import json
import numpy as np
import pytest
from faivor.model_metadata import ModelMetadata
from faivor.parse_data import ColumnMetadata, create_json_payloads
from faivor.run_docker import execute_model
from faivor.calculate_metrics import MetricsCalculator


MODEL_NAMES = ["pilot-model_1"]


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_execution(shared_datadir, model_name):
    model_dir = shared_datadir / "models"
    metadata_json = json.loads((model_dir / model_name / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    assert model_metadata.docker_image, "Docker image name should be provided"
    csv_path = model_dir / model_name / "data.csv"
    
    column_metadata_json = json.loads((model_dir / model_name / "column_metadata.json").read_text())
    column_metadata : list[ColumnMetadata] = ColumnMetadata.load_from_dict(column_metadata_json)
    
    inputs, _ = create_json_payloads(model_metadata, csv_path, column_metadata)
    try:
        prediction = execute_model(model_metadata, inputs)
    except Exception as e:
        raise RuntimeError(f"Model execution failed: {e}")

    assert prediction is not None, "Model execution should return a prediction."
    assert isinstance(prediction, list), "Prediction result should be a dictionary."

@pytest.fixture
def model_info(shared_datadir, model_name):
    """Fixture to provide model metadata and paths."""
    model_dir = shared_datadir / "models"
    model_path = model_dir / model_name
    
    if not model_path.exists():
        pytest.skip(f"Model directory {model_name} not found")
    
    metadata_json = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    
    model_type = metadata_json.get('model_type', 'classification')
    csv_path = model_path / "data.csv"
    column_metadata_path = model_path / "column_metadata.json"

    column_metadata_json = json.loads((column_metadata_path).read_text())
    column_metadata : list[ColumnMetadata] = ColumnMetadata.load_from_dict(column_metadata_json)
    
    
    return {
        "metadata_json": metadata_json,
        "model_metadata": model_metadata,
        "model_type": model_type,
        "csv_path": csv_path,
        "column_metadata_path": column_metadata_path,
        "column_metadata": column_metadata,
        "model_path": model_path,
        "model_name": model_name
    }

@pytest.fixture
def input_data(model_info):
    """Fixture to provide input and expected output data."""
    model_metadata = model_info["model_metadata"]
    csv_path = model_info["csv_path"]
    column_metadata = model_info["column_metadata"]

    inputs, expected_outputs = create_json_payloads(model_metadata, csv_path, column_metadata)
    return inputs, expected_outputs

@pytest.fixture
def model_predictions(model_info, input_data):
    """Fixture to execute model and get predictions."""
    model_metadata = model_info["model_metadata"]
    model_name = model_info["model_name"]
    inputs, _ = input_data
    
    try:
        predictions = execute_model(model_metadata, inputs)
        return predictions
    except Exception as e:
        pytest.fail(f"Model execution failed for {model_name}: {e}")

@pytest.fixture
def metrics_calculator(model_info, input_data, model_predictions) -> MetricsCalculator:
    """Fixture to provide configured metrics calculator."""
    model_metadata = model_info["model_metadata"]
    inputs, expected_outputs = input_data
    
    calculator : MetricsCalculator = MetricsCalculator(
        model_metadata=model_metadata,
        predictions=model_predictions,
        expected_outputs=expected_outputs,
        inputs=inputs
    )
    return calculator

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_metadata(model_info):
    """Test model metadata loading and validation."""
    model_metadata = model_info["model_metadata"]
    
    assert model_metadata.docker_image, "Docker image name should be provided"
    assert model_metadata.output, "Output field should be provided in model metadata"

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_data_loading(model_info, input_data):
    """Test loading and parsing input data."""
    inputs, expected_outputs = input_data
    
    assert len(inputs) > 0, "Should parse input samples from CSV"
    assert len(inputs) == len(expected_outputs), "Should have same number of inputs and expected outputs"

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_metrics_calculator_init(model_info, metrics_calculator):
    """Test metrics calculator initialization."""
    column_metadata_path = model_info["column_metadata_path"]
    
    if not column_metadata_path.exists():
        pytest.skip(f"Column metadata file not found")
    
    categorical_features = metrics_calculator.get_categorical_features_from_json(column_metadata_path)
    assert isinstance(categorical_features, list), "Should return a list of categorical features"

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_data_preparation(model_info, metrics_calculator, model_predictions):
    """Test preprocessing and validating prediction data."""
    model_name = model_info["model_name"]
    
    y_true, y_pred, valid_indices = metrics_calculator.prepare_data()
    
    assert len(y_true) > 0, f"Should have valid data points for {model_name}"
    assert len(y_true) == len(y_pred), "True values and predictions should have same length"
    assert len(valid_indices) <= len(model_predictions), "Valid indices should be subset of predictions"
    assert y_true.dtype.kind in 'fc', "True values should be numeric"
    assert y_pred.dtype.kind in 'fc', "Predicted values should be numeric"

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_output_format_analysis(model_info, metrics_calculator):
    """Test analyzing model output format."""
    model_type = model_info["model_type"]
    
    _, y_pred, _ = metrics_calculator.prepare_data()
    
    if model_type.lower() == "classification":
        # Check if values are within classification range
        assert np.all((np.isnan(y_pred) | ((y_pred >= 0) & (y_pred <= 1)))), \
            "Classification outputs should be in range [0,1] or NaN"

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_basic_metrics_calculation(model_info, metrics_calculator):
    """Test calculating standard performance metrics."""
    overall_metrics = metrics_calculator.calculate_metrics()
    
    assert overall_metrics, "Should return metrics dictionary"
    assert len(overall_metrics) > 0, "Should calculate at least one metric"

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_threshold_metrics_for_classification(model_info, metrics_calculator):
    """Test threshold-based metrics for classification models with probability outputs."""
    model_type = model_info["model_type"]
    
    if model_type.lower() != "classification":
        pytest.skip("Threshold metrics only apply to classification models")
    
    _, y_pred, _ = metrics_calculator.prepare_data()
    
    # check if we have valid prob outputs
    is_prob_output = (
        np.all((y_pred >= 0) & (y_pred <= 1)) and  
        not np.all(np.isin(y_pred, [0, 1]))        
    )
    
    if not is_prob_output:
        pytest.skip("Threshold metrics only apply to probability outputs")
    
    threshold_metrics = metrics_calculator.calculate_threshold_metrics()
    
    assert "roc_curve" in threshold_metrics, "Should contain ROC curve data"
    assert "pr_curve" in threshold_metrics, "Should contain PR curve data"
    assert "threshold_metrics" in threshold_metrics, "Should contain threshold metrics"
    
    # verify ROC
    roc = threshold_metrics['roc_curve']
    assert 'auc' in roc, "ROC data should include AUC score"
    assert 'fpr' in roc, "ROC data should include FPR values"
    assert 'tpr' in roc, "ROC data should include TPR values"
    assert 'thresholds' in roc, "ROC data should include thresholds"
    
    # verify PR 
    pr = threshold_metrics['pr_curve']
    assert 'average_precision' in pr, "PR data should include average precision"
    assert 'precision' in pr, "PR data should include precision values"
    assert 'recall' in pr, "PR data should include recall values"
    assert 'thresholds' in pr, "PR data should include thresholds"

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_all_metrics_calculation(model_info, metrics_calculator):
    """Test calculating complete metrics suite."""
    csv_path = model_info["csv_path"]
    column_metadata_path = model_info["column_metadata_path"]
    
    if not column_metadata_path.exists():
        pytest.skip("Column metadata file not found")
    
    all_metrics = metrics_calculator.calculate_all_metrics_from_json(csv_path, column_metadata_path)
    
    assert "model_info" in all_metrics, "Should contain model info"
    assert "overall" in all_metrics, "Should contain overall metrics"
    
    # check for subgroup analysis results if we have cat features
    categorical_features = metrics_calculator.get_categorical_features_from_json(column_metadata_path)
    if len(categorical_features) > 0:
        assert "subgroups" in all_metrics, "Should contain subgroup analysis"
        
        # verify there exists some subgroup analysis
        subgroups = all_metrics.get("subgroups", {})
        valid_subgroups = [f for f in categorical_features if f in subgroups]
        assert len(valid_subgroups) > 0, "At least one categorical feature should be analyzed"

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_metrics_json_output(shared_datadir, model_info, metrics_calculator):
    """Test saving metrics to JSON and validating file content."""
    model_name = model_info["model_name"]
    model_metadata = model_info["model_metadata"]
    csv_path = model_info["csv_path"]
    column_metadata_path = model_info["column_metadata_path"]
    
    if not column_metadata_path.exists():
        pytest.skip("Column metadata file not found")
    
    output_path = shared_datadir / f"test_metrics_output_{model_name}.json"
    
    try:
        _ = metrics_calculator.save_metrics_to_json_from_metadata(
            output_path, csv_path, column_metadata_path
        )
        
        assert output_path.exists(), f"JSON file should be created for {model_name}"
        
        loaded_metrics = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded_metrics["model_info"]["name"] == model_metadata.model_name, \
            "Saved JSON should contain correct model name"
        assert "overall" in loaded_metrics, "Saved JSON should contain overall metrics"
    finally:
        output_path.unlink(missing_ok=True)
