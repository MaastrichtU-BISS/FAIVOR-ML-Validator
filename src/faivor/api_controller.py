from fastapi import FastAPI, HTTPException

from faivor.docker import DockerMLPipeline
from faivor.model_validator import Model
app = FastAPI()
pipeline = DockerMLPipeline()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
def make_prediction(api_url: str, data: dict):
    """
    Send data to the model's REST API for prediction.

    Parameters
    ----------
    api_url : str
        URL of the model's REST API.
    data : dict
        Input data for prediction.

    Returns
    -------
    dict
        Prediction results.
    """
    try:
        result = pipeline.predict(api_url, data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate-statistics")
def calculate_statistics(dataset_path: str):
    """
    Calculate basic statistics for a dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset file.

    Returns
    -------
    dict
        Statistical summary of numeric columns.
    """
    try:
        stats = pipeline.calculate_statistics(dataset_path)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/stop-container")
def stop_container():
    """
    Stop the running Docker container.

    Returns
    -------
    dict
        Confirmation message.
    """
    try:
        if pipeline.stop_container():
            return {"message": "Container stopped successfully."}
        else:
            return {"message": "No container is currently running."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))