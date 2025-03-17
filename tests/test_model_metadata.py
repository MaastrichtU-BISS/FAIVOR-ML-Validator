import json
from faivor.model_metadata import ModelMetadata
from faivor.parse_data import create_json_payloads
from pathlib import Path
from typing import List


def test_model_metadata_creation(shared_datadir: Path):
    """Test the creation of ModelMetadata objects from JSON metadata files."""
    for model_location in get_model_paths(shared_datadir):
        model_metadata_path = model_location / "metadata.json"

        with open(model_metadata_path, "r", encoding="utf-8") as file:
            metadata_json = json.load(file)

        model_metadata = ModelMetadata(metadata_json)

        assert model_metadata.model_name is not None, "Model name should not be None"
        assert model_metadata.docker_image is not None, "Docker image should not be None"
        assert model_metadata.author is not None, "Author should not be None"
        assert model_metadata.contact_email is not None, "Contact email should not be None"
        assert isinstance(model_metadata.references, list), "References should be a list"
        assert len(model_metadata.inputs) > 0, "Inputs should not be empty"
        assert model_metadata.output != "", "Output should not be empty"


def test_create_json_payloads(shared_datadir):
    for model_location in get_model_paths(shared_datadir):
        model_metadata_path = model_location / "metadata.json"

        with open(model_metadata_path, "r", encoding="utf-8") as file:
            metadata_json = json.load(file)

        model_metadata = ModelMetadata(metadata_json)
        csv_path = model_location / "data.csv"
        inputs, outputs = create_json_payloads(model_metadata, csv_path)

        assert isinstance(inputs, list)
        assert isinstance(outputs, list)
        assert len(inputs) > 0
        assert len(outputs) > 0

        # Further checks can include verifying content correctness
        first_input = inputs[0]
        assert isinstance(first_input, dict)
        assert model_metadata.inputs[0]["description"] in inputs[0]
        assert model_metadata.output in outputs[0]

def get_model_paths(shared_datadir: Path) -> List[Path]:
    """Retrieve model paths from shared data directory."""
    model_dir = shared_datadir / "models"
    return [model_dir / subdir for subdir in model_dir.iterdir() if subdir.is_dir()]