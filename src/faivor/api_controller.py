from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import shutil
import json

from faivor.model_metadata import ModelMetadata
from faivor.parse_data import validate_csv_format

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/validate-csv/")
async def validate_csv(
    metadata_json: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Validate a CSV file against provided metadata and return parsed input/output payloads.

    Parameters
    ----------
    metadata_json : str
        JSON string containing metadata (inputs and output column).
    file : UploadFile
        CSV file uploaded by the user.

    Returns
    -------
    JSONResponse
        Parsed input and output payloads.
    """
    try:
        metadata_dict = json.loads(metadata_json)
        metadata = ModelMetadata(metadata_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {str(e)}")

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        columns = validate_csv_format(metadata, tmp_path)
        return JSONResponse(content={"csv_columns": columns})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")
    



@app.post("/evaluate")
async def evaluate():
    return {"message": "Model evaluation started"}


@app.get("/evaluation_status")
async def evaluation_status():
    return {"message": "Evaluation status"}

@app.get("/evaluation_result")
async def evaluation_result():
    return {"message": "Evaluation result"}