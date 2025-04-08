import json
import pandas as pd
from pathlib import Path
import csv
from typing import List, Dict, Any, Tuple


def detect_delimiter(csv_path: Path) -> str:
    """
    Detect the delimiter used in a CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    str
        The detected delimiter (either ';' or ',').
    """
    with open(csv_path, "r", encoding="utf-8") as file:
        sample = file.read(1024)  # Read a sample of the file
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter


def create_json_payloads(metadata: Any, csv_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create JSON payloads for input and output data based on provided metadata and CSV file.

    Parameters
    ----------
    metadata : ModelMetadata
        Parsed model metadata object containing input/output columns information.
    csv_path : Path
        Path to the CSV file containing data to be parsed.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        A tuple containing two elements:
        - inputs: List of dictionaries with input data for the model.
        - outputs: List of dictionaries containing the output values.
    """
    delimiter = detect_delimiter(csv_path)
    df = pd.read_csv(csv_path, sep=delimiter)

    input_columns = [input_feature["input_label"] for input_feature in metadata.inputs]
    output_column = metadata.output

    inputs = df[input_columns].to_dict(orient="records")
    outputs = df[[output_column]].to_dict(orient="records")

    return inputs, outputs
