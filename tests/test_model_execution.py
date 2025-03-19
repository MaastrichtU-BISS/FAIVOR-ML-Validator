import json
from faivor.model_metadata import ModelMetadata
from faivor.parse_data import create_json_payloads
from faivor.run_docker import execute_model

def test_model_execution(shared_datadir):
    model_dir = shared_datadir / "models"
    metadata_json = json.loads((model_dir / "pilot-model_1" / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    assert model_metadata.docker_image, "Docker image name should be provided"

    csv_path = model_dir / "pilot-model_1" / "data.csv"
    inputs, _ = create_json_payloads(model_metadata, csv_path)
    try:
        prediction = execute_model(model_metadata, inputs)
    except Exception as e:
        raise RuntimeError(f"Model execution failed: {e}")


    assert prediction is not None, "Model execution should return a prediction."
    assert isinstance(prediction, list), "Prediction result should be a dictionary."


    def test_model_metrics(shared_datadir):
        model_dir = shared_datadir / "models"
        metadata_json = json.loads((model_dir / "pilot-model_1" / "metadata.json").read_text(encoding="utf-8"))
        model_metadata = ModelMetadata(metadata_json)
        assert model_metadata.docker_image, "Docker image name should be provided"

        csv_path = model_dir / "pilot-model_1" / "data.csv"
        inputs, _ = create_json_payloads(model_metadata, csv_path)
        try:
            prediction = execute_model(model_metadata, inputs)
        except Exception as e:
            raise RuntimeError(f"Model execution failed: {e}")

        #TODO: compute_metric(prediction, expected_output)