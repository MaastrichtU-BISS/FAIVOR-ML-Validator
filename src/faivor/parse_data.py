from dataclasses import dataclass
import json
import pandas as pd
from pathlib import Path
import csv
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel

from faivor.model_metadata import ModelMetadata



@dataclass
class ColumnMetadata(BaseModel):
    """
    Represents metadata for  a single column in a CSV file.
    """
    id: str
    name_csv: str
    name_model: str
    categorical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ColumnMetadata instance to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the column metadata.
        """
        return {
            "id": self.id,
            "name_csv": self.name_csv,
            "name_model": self.name_model,
            "categorical": self.categorical,
        }
    
    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> list["ColumnMetadata"]:
        """
        Load column metadata from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing column metadata.

        Returns
        -------
        list[ColumnMetadata]
            List of ColumnMetadata instances.
        """
        return [
            ColumnMetadata(
                id=col["id"],
                name_csv=col["name_csv"],
                name_model=col.get("name_model", col["name_csv"]),
                categorical=col.get("categorical", False)
            ) for col in data.get("columns", [])
        ]




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
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
            return dialect.delimiter
    except (FileNotFoundError, IOError) as e:
        raise IOError(f"Could not open CSV file: {e}") from e
    

def load_csv(csv_path: Path) -> tuple[pd.DataFrame, List[str]]:
    """
    Load a CSV file into a DataFrame using the detected delimiter.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    tuple[pd.DataFrame, List[str]]
        - df: the loaded DataFrame
        - columns: list of all column names in the CSV

    Raises
    ------
    pd.errors.ParserError
        If the CSV cannot be parsed.
    """
    delimiter = detect_delimiter(csv_path)
    df = pd.read_csv(csv_path, sep=delimiter)
    return df, df.columns.tolist()


def validate_dataframe_format(
    metadata: ModelMetadata,
    csv_df: pd.DataFrame
) -> tuple[bool, str | None]:
    """
    Validate that all required input and output columns exist in the CSV.

    Parameters
    ----------
    metadata : ModelMetadata
        Object with `.inputs` (list of input_label) and `.output` (str).
    csv_df : pd.DataFrame
        DataFrame containing the CSV data.

    Returns
    -------
    tuple[bool, str | None]
        (True, None) if all required columns are present,
        (False, message) otherwise.
    """

    required = [inp.input_label for inp in metadata.inputs] + [metadata.output]
    missing = [col for col in required if col not in csv_df.columns]

    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"

    return True, None


def create_json_payloads(
    metadata: ModelMetadata,
    csv_path: Path,
    column_metadata: list[ColumnMetadata]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create JSON payloads for input and output data based on provided metadata and CSV file.

    This function reuses `validate_dataframe_format` to ensure the CSV is valid before extracting payloads.

    Parameters
    ----------
    metadata : ModelMetadata
        Parsed model metadata with `.inputs` (list of {"input_label": ...})
        and `.output` (str).
    csv_path : Path
        Path to the CSV file.
    column_metadata : list[ColumnMetadata]
        Additional metadata for each column, used to rename columns in the DataFrame.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        - inputs: list of dicts for model inputs
        - outputs: list of dicts for model outputs

    Raises
    ------
    ValueError
        If the CSV is missing required columns (propagated from `validate_csv_format`).
    """

    df, all_columns = load_csv(csv_path)

    if column_metadata:
        # If data metadata is provided, rename columns in the DataFrame that mates csv_name to model_name
        for col in column_metadata:
            if col.name_csv in df.columns:
                df.rename(columns={col.name_csv: col.name_model}, inplace=True)
                
    is_valid, message = validate_dataframe_format(metadata, df)

    if not is_valid:
        raise ValueError(f"CSV format validation failed: {message}")

    input_cols = [inp.input_label for inp in metadata.inputs]
    output_col = metadata.output

    inputs = df[input_cols].to_dict(orient="records")
    outputs = df[[output_col]].to_dict(orient="records")

    return inputs, outputs