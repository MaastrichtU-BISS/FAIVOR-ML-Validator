import json
from faivor.model_metadata import ModelMetadata


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
