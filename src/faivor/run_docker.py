import docker
import requests
import time
from typing import Dict, Any, Tuple


def run_docker_container(image_name: str, port: int = 8000) -> docker.models.containers.Container:
    """
    Run a Docker container from the specified image.

    Parameters
    ----------
    image_name : str
        Name of the Docker image to run.
    port : int, optional
        Port on which the container's API will be exposed, by default 8000.

    Returns
    -------
    docker.models.containers.Container
        The running Docker container instance.
    """
    try:
        client = docker.from_env()
        client.ping()  # Test connection to Docker daemon
    except docker.errors.DockerException as e:
        raise RuntimeError(f"Cannot connect to Docker daemon: {e}")

    container = client.containers.run(
        image=image_name,
        detach=True,
        remove=True,
        ports={"8000/tcp": port}
    )
    time.sleep(2)
    return container


def execute_model(metadata: Any, input_payload: Dict[str, Any], port: int = 8000) -> Any:
    """
    Execute the model using provided input data.

    Parameters
    ----------
    metadata : ModelMetadata
        Parsed metadata object containing Docker image information.
    input_payload : Dict[str, Any]
        JSON payload with input data to send to the model.

    Returns
    -------
    Any
        The prediction result returned by the model.
    """
    container = run_docker_container(metadata.docker_image, port)

    try:
        response = requests.post(f"http://localhost:{port}/predict", json=input_payload)
        response.raise_for_status()
        result = response.json()
    finally:
        container.stop()

    return result