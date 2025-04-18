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


def validate_csv(
    metadata: Any,
    csv_path: Path
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate that all required input and output columns exist in the CSV.

    Parameters
    ----------
    metadata : Any
        Object with `.inputs` (list of dicts with "input_label") and `.output` (str).
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - df: the loaded DataFrame
        - columns: list of all column names in the CSV

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    df = load_csv(csv_path)

    required = [inp["input_label"] for inp in metadata.inputs] + [metadata.output]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    return df, df.columns.tolist()


def create_json_payloads(
    metadata: Any,
    csv_path: Path
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create JSON payloads for input and output data based on provided metadata and CSV file.

    This function reuses `validate_csv` to ensure the CSV is valid before extracting payloads.

    Parameters
    ----------
    metadata : Any
        Parsed model metadata with `.inputs` (list of {"input_label": ...})
        and `.output` (str).
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        - inputs: list of dicts for model inputs
        - outputs: list of dicts for model outputs

    Raises
    ------
    ValueError
        If the CSV is missing required columns (propagated from `validate_csv`).
    """
    df, all_columns = validate_csv(metadata, csv_path)

    input_cols = [inp["input_label"] for inp in metadata.inputs]
    output_col = metadata.output

    inputs = df[input_cols].to_dict(orient="records")
    outputs = df[[output_col]].to_dict(orient="records")

    return inputs, outputs