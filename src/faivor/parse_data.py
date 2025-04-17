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
        The detected delimiter (comma, semicolon, tab, or pipe).

    Raises
    ------
    IOError
        If the file cannot be opened.
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            sample = f.read(1024)
            dialect = csv.Sniffer().sniff(sample, delimiters=";, \t|")
            return dialect.delimiter
    except (FileNotFoundError, IOError) as e:
        raise IOError(f"Could not open CSV file: {e}") from e
    

def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame using the detected delimiter.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded data.

    Raises
    ------
    pd.errors.ParserError
        If the CSV cannot be parsed.
    """
    delimiter = detect_delimiter(csv_path)
    return pd.read_csv(csv_path, sep=delimiter)


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


