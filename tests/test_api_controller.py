from fastapi.testclient import TestClient
from pathlib import Path
import json

import pytest
from faivor.api_controller import app

MODEL_NAMES = ["pilot-model_1", "pilot-model_2"]

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
