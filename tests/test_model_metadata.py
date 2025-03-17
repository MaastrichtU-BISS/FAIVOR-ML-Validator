import json
from faivor.model_metadata import ModelMetadata
from faivor.parse_data import create_json_payloads


def test_model_metadata_creation(shared_datadir):
    # Load model metadata from a JSON file in the shared data directory
    model_metadata_path = shared_datadir / "pilot-model_2" / "metadata.json"

    with open(model_metadata_path, "r", encoding="utf-8") as file:
        metadata_json = json.load(file)

    model_metadata: ModelMetadata = ModelMetadata(metadata_json)

    # Test real expected values
    assert (
        model_metadata.model_name
        == "Prediction model for tube feeding dependency during chemoradiotherapy for at least four weeks in head and neck cancer patients"
    )
    assert model_metadata.docker_image == "jvsoest/willemsen_tubefeed"
    assert model_metadata.author == "Willemsen A.C.H. et al."
    assert model_metadata.contact_email == "j.vansoest@maastrichtuniversity.nl"
    assert len(model_metadata.references) == 1
    assert model_metadata.references[0] == "https://doi.org/10.1016/j.clnu.2019.11.033"
    assert len(model_metadata.inputs) > 0
    assert model_metadata.output == "Tube feeding of patient"


def test_create_json_payloads(shared_datadir):
    metadata_json = json.loads((shared_datadir / "pilot-model_2" / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    csv_path = shared_datadir / "pilot-model_2" / "data.csv"
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