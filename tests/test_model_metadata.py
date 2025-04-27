import json

import pandas as pd
import pytest
from faivor.model_metadata import ModelInput, ModelMetadata
from faivor.parse_data import (
    detect_delimiter,
    load_csv,
    validate_csv_format,
    create_json_payloads,
)
from pathlib import Path
from typing import List



def test_model_metadata_creation(shared_datadir: Path):
    """ModelMetadata must populate all required fields."""
    for model_location in get_model_paths(shared_datadir):
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
    for model_location in get_model_paths(shared_datadir):
        metadata_content = json.loads((model_location / "metadata.json").read_text())
        metadata = ModelMetadata(metadata_content)
        csv_path = model_location / "data.csv"

        # validate_csv_format returns df, columns
        df, cols = validate_csv_format(metadata, csv_path)
        # all required present
        for req in [inp.input_label for inp in metadata.inputs] + [metadata.output]:
            assert req in cols

        inputs, outputs = create_json_payloads(metadata, csv_path)
        assert isinstance(inputs, list) and inputs, "inputs empty or wrong type"
        assert isinstance(outputs, list) and outputs, "outputs empty or wrong type"
        # spot‑check that keys match labels
        assert set(inputs[0].keys()) == {inp.input_label for inp in metadata.inputs}
        assert set(outputs[0].keys()) == {metadata.output}


def test_validate_csv_format_missing_column(shared_datadir : Path, tmp_path: Path):
    """validate_csv_format raises ValueError listing missing columns."""
    # create a minimal metadata expecting 'a','b' inputs and 'c' output
    metadata_content = json.loads((shared_datadir / "models" / "pilot-model_1" / "metadata.json").read_text(encoding="utf-8"))
    metadata = ModelMetadata(metadata_content)
    metadata.inputs.clear()
    metadata.inputs.append(ModelInput("a"))
    metadata.inputs.append(ModelInput("b"))
    metadata.inputs.append(ModelInput("c"))
    # write CSV with only 'a' and 'c'
    p = tmp_path / "t.csv"
    p.write_text("a,c\n1,2\n3,4\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        validate_csv_format(metadata, p)
    msg = str(exc.value)
    assert "Missing required columns" in msg
    assert "b" in msg  # missing input
    # should not mention 'c', since it's present
    assert "c" not in msg.split(":" )[1]


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


def test_load_csv_and_roundtrip(tmp_path: Path):
    """load_csv → DataFrame, then back to CSV yields same columns."""
    content = "foo,bar\n10,20\n"
    p = tmp_path / "test.csv"
    p.write_text(content, encoding="utf-8")
    df = load_csv(p)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["foo", "bar"]


def get_model_paths(shared_datadir: Path) -> List[Path]:
    """Retrieve model paths from shared data directory."""
    model_dir = shared_datadir / "models"
    # Return only the first model
    # return [model_dir / "pilot-model_1"]
    return [model_dir / subdir for subdir in model_dir.iterdir() if subdir.is_dir()]