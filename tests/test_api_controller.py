from fastapi.testclient import TestClient
import pytest
from httpx import AsyncClient
from pathlib import Path
import json
from faivor.api_controller import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_validate_csv(shared_datadir: Path):
    model_dir = shared_datadir / "models"
    metadata_path = model_dir / "pilot-model_1" / "metadata.json"
    csv_path = model_dir / "pilot-model_1" / "data.csv"

    # load and stringify metadata
    metadata_dict = json.load(open(metadata_path))
    metadata_str = json.dumps(metadata_dict)

    # upload
    with open(csv_path, "rb") as csv_file:
        files = {
            "metadata_json": (None, metadata_str),
            "file": ("data.csv", csv_file, "text/csv"),
        }
        response = client.post("/validate-csv/", files=files)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("inputs"), list)
    assert isinstance(data.get("outputs"), list)
