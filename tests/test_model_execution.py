import json
from faivor.model_metadata import ModelMetadata
from faivor.parse_data import create_json_payloads
from faivor.run_docker import execute_model


def test_model_execution(shared_datadir):
    metadata_json = json.loads((shared_datadir / "pilot-model_2" / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    assert model_metadata.docker_image, "Docker image name should be provided"

    csv_path = shared_datadir / "pilot-model_2" / "data.csv"
    inputs, _ = create_json_payloads(model_metadata, csv_path)

    prediction = execute_model(model_metadata, inputs[0])

    assert prediction is not None, "Model execution should return a prediction."
    assert isinstance(prediction, dict), "Prediction result should be a dictionary."