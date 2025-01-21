from fastapi import FastAPI, HTTPException
import docker
import requests
import statistics
import pandas as pd

app = FastAPI()

class DockerMLPipeline:
    """
    A class to manage Docker containers for running ML models, pulling datasets, making predictions, and performing statistics.
    """

    def __init__(self):
        """
        Initializes the DockerMLPipeline class.

        Creates a Docker client for managing containers.

        Raises
        ------
        docker.errors.DockerException
            If the Docker client cannot be initialized.
        """
        self.client = docker.from_env()
        self.container = None

    def pull_docker_image(self, image_name):
        """
        Pulls a Docker image from the Docker registry.

        Parameters
        ----------
        image_name : str
            Name of the Docker image to pull.

        Returns
        -------
        bool
            True if the image was pulled successfully.

        Raises
        ------
        docker.errors.ImageNotFound
            If the specified image is not found.
        docker.errors.APIError
            If there is an error during the pull operation.
        """
        self.client.images.pull(image_name)
        return True

    def start_container(self, image_name, ports):
        """
        Starts a Docker container from a specified image.

        Parameters
        ----------
        image_name : str
            Name of the Docker image to run.
        ports : dict
            Mapping of container ports to host ports (e.g., {"5000/tcp": 5000}).

        Returns
        -------
        str
            ID of the running container.

        Raises
        ------
        docker.errors.ContainerError
            If the container fails to start.
        docker.errors.ImageNotFound
            If the image does not exist.
        docker.errors.APIError
            If there is an error starting the container.
        """
        self.container = self.client.containers.run(image_name, detach=True, ports=ports)
        return self.container.id

    def stop_container(self):
        """
        Stops the currently running Docker container.

        Returns
        -------
        bool
            True if the container was stopped successfully.

        Raises
        ------
        docker.errors.APIError
            If there is an error stopping the container.
        """
        if self.container:
            self.container.stop()
            self.container = None
            return True
        return False

    def predict(self, api_url, data):
        """
        Sends data to the model's REST API for prediction.

        Parameters
        ----------
        api_url : str
            URL of the model's REST API endpoint (e.g., "http://localhost:5000/predict").
        data : dict
            Input data for the prediction in JSON format.

        Returns
        -------
        dict
            The prediction results returned by the API.

        Raises
        ------
        requests.exceptions.RequestException
            If there is an error with the HTTP request.
        """
        response = requests.post(api_url, json=data)
        response.raise_for_status()
        return response.json()

    def calculate_statistics(self, dataset_path):
        """
        Calculates basic statistics on a dataset.

        Parameters
        ----------
        dataset_path : str
            Path to the dataset file (e.g., CSV file).

        Returns
        -------
        dict
            A dictionary containing mean, median, and standard deviation for numeric columns.

        Raises
        ------
        FileNotFoundError
            If the dataset file is not found.
        ValueError
            If the dataset contains no numeric columns.
        """
        df = pd.read_csv(dataset_path)
        numeric_cols = df.select_dtypes(include=['number'])

        if numeric_cols.empty:
            raise ValueError("No numeric columns found in the dataset.")

        stats = {
            col: {
                "mean": numeric_cols[col].mean(),
                "median": numeric_cols[col].median(),
                "std_dev": numeric_cols[col].std()
            } for col in numeric_cols
        }
        return stats