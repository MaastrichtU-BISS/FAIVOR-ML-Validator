import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
# pylint: disable=no-name-in-module
from pydantic import BaseModel

from faivor.model_metadata import ModelMetadata
from faivor.parse_data import validate_csv_format

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


class ColumnsResponse(BaseModel):
    csv_columns: list[str]
    model_input_columns: list[str]



class ModelMetrics(BaseModel):
    model_name: str
    metrics: dict[str, float]


@app.post(
    "/validate-csv/",
    response_model=ColumnsResponse,
    summary="Validate CSV against model metadata",
    description=(
        "Uploads a FAIR model metadata JSON string and a CSV file, "
        "verifies that all required columns are present, "
        "and returns the list of CSV column names."
    ),
    responses={
        400: {
            "description": "Invalid metadata or CSV format",
            "content": {
                "application/json": {
                    "example": {"detail": "Missing required columns: foo, bar"}
                }
            },
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
        raise HTTPException(400, f"Invalid metadata JSON: {e}")

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(csv_file.file, tmp)
            tmp_path = Path(tmp.name)

        _, columns = validate_csv_format(metadata, tmp_path)

        return JSONResponse(content={"csv_columns": columns, "model_input_columns": [input_obj.input_label for input_obj in metadata.inputs]})
    except ValueError as e:
        # raised by validate_csv_format for missing columns
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Failed to process CSV: {e}")


@app.post(
    "/validate-model",
    response_model=ModelMetrics,
    summary="Validate ML model",
    description=(
        "Validates the ML model by checking the ML FAIR metadata, pulling the specified docker image and running metrics on the predictions derived from the provided CSV file. The CSV file should contain the same columns as the model metadata inputs as well as the expected predictions. "
        "Returns the model name and metrics."
    )
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
    return {"message": "Model evaluation started"}
