import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# pylint: disable=no-name-in-module
import pandas as pd
from pydantic import BaseModel

from faivor.model_metadata import ModelMetadata
from faivor.parse_data import create_json_payloads, load_csv, validate_dataframe_format

app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome "}


class ValidationResponse(BaseModel):
    valid: bool
    message: str | None = None
    csv_columns: list[str]
    model_input_columns: list[str]



class ModelMetrics(BaseModel):
    model_name: str
    metrics: dict[str, float]


@app.post(
    "/validate-csv/",
    response_model=ValidationResponse,
    summary="Validate CSV against model metadata",
    description=(
        "Uploads a FAIR model metadata JSON string and a CSV file, "
        "verifies that all required columns are present, "
        "and returns the list of CSV column names."
    ),
    responses={
        400: {
            "description": "Invalid metadata JSON or CSV format",
            "content": {
                "application/json": {
                    "example": {"message": "Invalid metadata JSON: ..."}
                }
            }
        },
    },
)
async def validate_csv(
    model_metadata: str = Form(
        ...,
        description="FAIR model metadata JSON, containing `inputs` (list of `{input_label}`) and `output` field",
    ),
    csv_file: UploadFile = File(
        ..., description="CSV file to validate; delimiter is auto-detected"
    ),
) -> JSONResponse:
    """
    Validate a CSV file against provided metadata and return all CSV columns.
    """
    try:
        md = json.loads(model_metadata)
        metadata = ModelMetadata(md)
    except Exception as e:
        raise HTTPException(400, f"Invalid metadata JSON: {e}") from e

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(csv_file.file, tmp)
            tmp_path = Path(tmp.name)

        df_data, columns = load_csv(tmp_path)
    except pd.errors.ParserError as e:
        raise HTTPException(400, f"Invalid CSV format: {e}") from e

    validation_result, msg = validate_dataframe_format(metadata, df_data)

    return JSONResponse(
        content={
            "valid": validation_result,
            "message": msg,
            "csv_columns": columns,
            "model_input_columns": [
                input_obj.input_label for input_obj in metadata.inputs
            ],
        }
    )

@app.post(
    "/validate-model",
    response_model=ModelMetrics,
    summary="Validate ML model",
    description=(
        "Validates the ML model by checking the ML FAIR metadata, pulling the specified docker image and running metrics on the predictions derived from the provided CSV file. The CSV file should contain the same columns as the model metadata inputs as well as the expected predictions. "
        "Returns the model name and metrics."
    ),
)
async def validate_model(
    model_metadata: str = Form(
        ...,
        description="FAIR model metadata JSON, containing `inputs` (list of `{input_label}`) and `output` field",
    ),
    csv_file: UploadFile = File(
        ..., description="CSV file to validate; delimiter is auto-detected"
    ),
    data_metadata: str = Form(
        ...,
        description="Metadata JSON, containing naming ",
    ),
) -> JSONResponse:
    """
    Validate a model with metadata and CSV data.
    """
    try:
        # Parse the model metadata to extract model name
        md = json.loads(model_metadata)
        model_name = md.get('model_name', md.get('name', 'unknown_model'))

        # Parse data metadata if provided
        try:
            data_md = json.loads(data_metadata) if data_metadata else {}
        except json.JSONDecodeError:
            data_md = {}

        # TODO: Implement actual model validation logic
        # For now, return a basic response with placeholder metrics
        # inputs, _ = create_json_payloads(model_metadata, csv_file)
        # prediction = execute_model(model_metadata, inputs)

        # Return proper response format matching ModelMetrics schema
        return JSONResponse(
            content={
                "model_name": model_name,
                "metrics": {
                    "validation_status": 1.0,
                    "data_processed": 1.0
                }
            }
        )

    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid metadata JSON: {e}") from e
    except Exception as e:
        raise HTTPException(500, f"Model validation failed: {e}") from e
