import json

import pandas as pd
import pytest
from faivor.model_metadata import ModelInput, ModelMetadata
from faivor.parse_data import (
    ColumnMetadata,
    detect_delimiter,
    load_csv,
    validate_dataframe_format,
    create_json_payloads,
)
from pathlib import Path
from typing import List


MODEL_NAMES = ["pilot-model_1", "pilot-model_2"]

def get_all_model_paths(shared_datadir: Path) -> List[Path]:
    """Retrieve model paths from shared data directory."""
    model_dir = shared_datadir / "models"
    return [model_dir / subdir for subdir in model_dir.iterdir() if subdir.is_dir()]

def test_model_metadata_creation(shared_datadir: Path):
    """ModelMetadata must populate all required fields."""
    for model_location in get_all_model_paths(shared_datadir):
        metadata_content = json.loads((model_location / "metadata.json").read_text(encoding="utf-8"))
        metadata = ModelMetadata(metadata_content)
        assert metadata.model_name, "model_name empty"
        assert metadata.docker_image, "docker_image empty"
        assert metadata.author, "author empty"
        assert metadata.contact_email, "contact_email empty"
        assert isinstance(metadata.references, list), "references not list"
        assert metadata.inputs, "inputs empty"
        assert metadata.output, "output empty"


def test_create_json_payloads_and_validate(shared_datadir: Path):
    """CSV → payloads round‑trip and format validation."""
    for model_location in get_all_model_paths(shared_datadir):
        metadata_content = json.loads((model_location / "metadata.json").read_text())
        metadata = ModelMetadata(metadata_content)
        csv_path = model_location / "data.csv"

        column_metadata_json = json.loads((model_location / "column_metadata.json").read_text())
        columns_metadata : list[ColumnMetadata] = ColumnMetadata.load_from_dict(column_metadata_json)

        inputs, outputs = create_json_payloads(metadata, csv_path, columns_metadata)
        assert isinstance(inputs, list) and inputs, "inputs empty or wrong type"
        assert isinstance(outputs, list) and outputs, "outputs empty or wrong type"
        # spot‑check that keys match labels
        assert set(inputs[0].keys()) == {inp.input_label for inp in metadata.inputs}
        assert set(outputs[0].keys()) == {metadata.output}

def test_failing_csv(shared_datadir: Path):
    """CSV → payloads round‑trip and format validation."""
    model_dir = shared_datadir / "models"
    metadata_json = json.loads((model_dir / "pilot-model_1" / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    assert model_metadata.docker_image, "Docker image name should be provided"
    csv_path = model_dir / "pilot-model_1" / "changed_data" / "data.csv"

    with pytest.raises(ValueError):
        inputs, outputs = create_json_payloads(model_metadata, csv_path, [])

def test_changed_csv(shared_datadir: Path):
    """CSV → payloads round‑trip and format validation."""
    model_dir = shared_datadir / "models"
    metadata_json = json.loads((model_dir / "pilot-model_1" / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    assert model_metadata.docker_image, "Docker image name should be provided"
    csv_path = model_dir / "pilot-model_1" / "changed_data" / "data.csv"

    column_metadata_json = json.loads((model_dir / "pilot-model_1" / "changed_data" / "column_metadata.json").read_text(encoding="utf-8"))
    columns_metadata : list[ColumnMetadata] = ColumnMetadata.load_from_dict(column_metadata_json)

    inputs, outputs = create_json_payloads(model_metadata, csv_path, columns_metadata)
    assert isinstance(inputs, list) and inputs, "inputs empty or wrong type"
    assert isinstance(outputs, list) and outputs, "outputs empty or wrong type"
    # spot‑check that keys match labels
    assert set(inputs[0].keys()) == {inp.input_label for inp in model_metadata.inputs}
    assert set(outputs[0].keys()) == {model_metadata.output}


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_validate_dataframe_format_missing_column(shared_datadir: Path, model_name: str):
    """validate_dataframe_format raises ValueError listing missing columns."""
    # create a minimal metadata expecting 'a','b' inputs and 'c' output
    metadata_content = json.loads((shared_datadir / "models" / model_name / "metadata.json").read_text(encoding="utf-8"))
    metadata = ModelMetadata(metadata_content)
    metadata.inputs.clear()
    metadata.inputs.append(ModelInput("a"))
    metadata.inputs.append(ModelInput("b"))
    metadata.inputs.append(ModelInput("c"))
    
    # create a DataFrame with only 'a' and 'c'
    df = pd.DataFrame({"a": [1, 3], "c": [2, 4]})

    valid, msg = validate_dataframe_format(metadata, df)
    assert not valid
    assert "Missing required columns" in msg
    assert "b" in msg  # missing input
    # should not mention 'c', since it's present
    assert "c" not in msg.split(":")[1]


@pytest.mark.parametrize("content,expected", [
    ("x,y\n1,2", ","),          # comma
    ("x;y\n1;2", ";"),          # semicolon
    ("x\ty\n1\t2", "\t"),       # tab
    ("x|y\n1|2", "|"),          # pipe
])
def test_detect_delimiter_various(tmp_path: Path, content: str, expected: str):
    """detect_delimiter should sniff , ; \\t and | correctly."""
    p = tmp_path / "d.csv"
    p.write_text(content, encoding="utf-8")
    assert detect_delimiter(p) == expected

def test_load_csv(tmp_path: Path):
    """load_csv should return DataFrame."""
    content = "foo,bar\n10,20\n"
    p = tmp_path / "test.csv"
    p.write_text(content, encoding="utf-8")
    df, columns = load_csv(p)
    assert isinstance(df, pd.DataFrame)
    assert list(columns) == ["foo", "bar"]

def test_load_csv_and_roundtrip(tmp_path: Path):
    """load_csv → DataFrame, then back to CSV yields same columns."""
    content = "foo,bar\n10,20\n"
    p = tmp_path / "test.csv"
    p.write_text(content, encoding="utf-8")
    df, columns = load_csv(p)
    assert isinstance(df, pd.DataFrame)
    assert list(columns) == ["foo", "bar"]
