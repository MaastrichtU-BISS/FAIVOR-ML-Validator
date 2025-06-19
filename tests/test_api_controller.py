from fastapi.testclient import TestClient
from pathlib import Path
import json

import pytest
from faivor.api_controller import app

MODEL_NAMES = ["pilot-model_1"]

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_validate_csv_format(shared_datadir: Path, model_name: str):
    model_dir = shared_datadir / "models"
    metadata_path = model_dir / model_name / "metadata.json"
    csv_path = model_dir / model_name / "data.csv"

    # load and stringify metadata
    metadata_dict = json.load(open(metadata_path))
    metadata_str = json.dumps(metadata_dict)

    # upload
    with open(csv_path, "rb") as csv_file:
        files = {
            "model_metadata": (None, metadata_str),
            "csv_file":      ("data.csv", csv_file, "text/csv"),
        }
        response = client.post("/validate-csv/", files=files)


    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("csv_columns"), list)
    assert isinstance(data.get("model_input_columns"), list)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_validate_model(shared_datadir: Path, model_name: str):
    """
    Test the validate_model endpoint.
    """
    model_dir = shared_datadir / "models"
    metadata_path = model_dir / model_name / "metadata.json"
    csv_path = model_dir / model_name / "data.csv"
    column_metadata_path = model_dir / model_name / "column_metadata.json"

    # Load and stringify metadata
    metadata_dict = json.load(open(metadata_path))
    metadata_str = json.dumps(metadata_dict)

    # Load data metadata if available
    data_metadata_dict = {}
    if column_metadata_path.exists():
        data_metadata_dict = json.load(open(column_metadata_path))
    data_metadata_str = json.dumps(data_metadata_dict)

    # Upload files
    with open(csv_path, "rb") as csv_file:
        files = {
            "model_metadata": (None, metadata_str),
            "csv_file": ("data.csv", csv_file, "text/csv"),
            "data_metadata": (None, data_metadata_str),
        }
        response = client.post("/validate-model", files=files)

    # Validate response
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert "model_name" in data, "Response should contain model_name"
    assert "metrics" in data, "Response should contain metrics"
    assert "overall_metrics" in data["metrics"], "Metrics should include overall_metrics"
    assert "threshold_metrics" in data["metrics"], "Metrics should include threshold_metrics"