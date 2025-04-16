from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/evaluate")
async def evaluate():
    return {"message": "Model evaluation started"}


@app.get("/evaluation_status")
async def evaluation_status():
    return {"message": "Evaluation status"}

@app.get("/evaluation_result")
async def evaluation_result():
    return {"message": "Evaluation result"}