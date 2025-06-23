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
from pyparsing import Optional

from faivor.calculate_metrics import MetricsCalculator
from faivor.metrics.classification import classification_metrics
from faivor.metrics.regression import regression_metrics
from faivor.model_metadata import ModelMetadata
from faivor.parse_data import ColumnMetadata, create_json_payloads, load_csv, validate_dataframe_format
from faivor.run_docker import execute_model

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


class MetricDescription(BaseModel):
    name: str
    description: str
    type: str


@app.post(
    "/retrieve-metrics",
    response_model=list[MetricDescription],
    summary="Retrieve applicable metrics for the model",
    description=(
        "Returns a list of MetricDescription objects containing the names and descriptions "
        "of metrics applicable to the model based on the provided FAIR model metadata. "
        "Optionally filter by category (`performance`, `fairness`, `explainability`)."
    ),
)
async def retrieve_metrics(
    model_metadata: str = Form(
        ...,
        description="FAIR model metadata JSON, containing `inputs` (list of `{input_label}`) and `output` field",
    ),
) -> list[MetricDescription]:
    """
    Retrieve applicable metrics for the model.
    """
    try:
        # Parse the model metadata
        md = json.loads(model_metadata)
        metadata = ModelMetadata(md)
        model_type = metadata.metadata.get("model_type", "regression").lower()

        # detect appropriate metrics module based on model type
        metrics_module = (classification_metrics if model_type.lower() == "classification" 
                        else regression_metrics)

        # Collect metrics
        performance_metrics = [
                MetricDescription(
                    name=metric.regular_name, description=metric.description, type = "performance"
                )
                for metric in metrics_module.PERFORMANCE_METRICS
            ]
        fairness_metrics = [
                MetricDescription(
                    name=metric.regular_name, description=metric.description, type = "fairness"
                )
                for metric in metrics_module.FAIRNESS_METRICS
            ]
        explainability_metrics= [
                MetricDescription(
                    name=metric.regular_name, description=metric.description, type = "explainability"
                )
                for metric in metrics_module.EXPLAINABILITY_METRICS
            ]


        # Return all metrics as a flat list
        return performance_metrics + fairness_metrics + explainability_metrics

    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid metadata JSON: {e}") from e
    except Exception as e:
        raise HTTPException(500, f"Failed to retrieve metrics: {e}") from e

class ColumnMetadataStruct(BaseModel):
    columns: list[ColumnMetadata]
    
class ModelMetrics(BaseModel):
    model_name: str
    metrics: dict[str, float]



@app.post(
    "/validate-model",
    summary="Validate ML model",
    description=(
        "Validates the ML model by checking the provided FAIR model metadata, "
        "the CSV file, and the column metadata. "
        "The `column_metadata` input must follow the specified format."
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
    column_metadata: ColumnMetadataStruct | None = Form(
        None,
        description="Metadata JSON, containing naming mapping for the CSV columns, as well as information about whether the column is categorical (it can be used for threshold metrics calculation) or not. ",
    ),
) -> JSONResponse:
    """
    Validate a model with metadata and CSV data.
    """
    try:
        # Parse the model metadata
        md = json.loads(model_metadata)
        metadata = ModelMetadata(md)
        model_name = metadata.model_name or "unknown_model"

        # Parse column metadata if provided
        try:
            if column_metadata:
                col_metadata = json.loads(column_metadata) if column_metadata else {}
                colomns_metadata : list[ColumnMetadata] = ColumnMetadata.load_from_dict(col_metadata)
            else:
                colomns_metadata = []
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "Invalid column metadata JSON format.") from exc

        # Load CSV data
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(csv_file.file, tmp)
            tmp_path = Path(tmp.name)


        # Prepare inputs and expected outputs
        inputs, expected_outputs = create_json_payloads(metadata, tmp_path, colomns_metadata)

        # Execute model and get predictions
        try:
            predictions = execute_model(metadata, inputs)
        except Exception as e:
            raise HTTPException(500, f"Model execution failed: {e}") from e

        # Initialize MetricsCalculator
        metrics_calculator = MetricsCalculator(
            model_metadata=metadata,
            predictions=predictions,
            expected_outputs=expected_outputs,
            inputs=inputs,
        )

        # Calculate metrics
        overall_metrics = metrics_calculator.calculate_metrics()

        # Calculate threshold metrics (if applicable)
        threshold_metrics = {}
        if metadata.metadata.get("model_type", "").lower() == "classification":
            try:
                threshold_metrics = metrics_calculator.calculate_threshold_metrics()
            except Exception as e:
                threshold_metrics = {
                    "status": "error",
                    "message": f"Failed to calculate threshold metrics: {e}",
                }

        # Combine results
        metrics = {
            "overall_metrics": overall_metrics,
            "threshold_metrics": threshold_metrics,
        }

        return JSONResponse(
            content={
                "model_name": model_name,
                "metrics": metrics,
            }
        )

    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid metadata JSON: {e}") from e
    except Exception as e:
        raise HTTPException(500, f"Model validation failed: {e}") from e
