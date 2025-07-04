# FAIRmodels Validator

Validator library for ML models (FAIRmodels).

![](https://img.shields.io/badge/python-3.11+-blue.svg)
![GitHub license](https://img.shields.io/github/license/MaastrichtU-BISS/FAIVOR-backend)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

This is the backend of the FAIRmodels-validator project. It is a library that validates FAIR models.

It is a REST API server built with FastAPI.
The architecture of the project is shown in the following diagram:

![techstack](./architecture.drawio.png)

## Local development

### Environment setup and service run

1. **Clone the Repository**  
   Clone the project repository to your local machine.

2. **Install Dependencies**  
   In the root directory of the repository, run the following command to install dependencies (Python 3.11+ and Poetry 1.0+ required):

   `poetry install`

3. **Activate the Virtual Environment**  
   Activate the virtual environment created by Poetry with:

   `poetry shell`

4. Juptyer Notebook Kernel  
   To use this environment in Jupyter notebooks, install a custom kernel by running:

   `python -m ipykernel install --user --name=faivor-ml-validator --display-name "FAIVOR-ML-Validator"`

   This command makes the environment available in Jupyter Notebook under the kernel name **FAIVOR-ML-Validator**.

### Run the REST API server

To run the REST API server, execute the following command:

```bash
uvicorn src.FAIRmodels-validator.api_controller:app --reload
```

The server will be running on [http://localhost:8000](http://localhost:8000). You can access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

## Docker Requirements

**Important:** This service requires Docker access to validate ML models. When running in a container, you must mount the Docker socket. See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.

## Local development

To get started with this project, please follow these steps to set up the environment and configure Jupyter for notebook use:
