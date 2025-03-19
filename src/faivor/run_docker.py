import docker
import requests
import socket
import time
from contextlib import closing
from typing import Dict, Any, Tuple


# Mapping of status codes to strings
status_map = {
    0: "No prediction requested",
    1: "Prediction requested",
    2: "Prediction in progress",
    3: "Prediction completed",
    4: "Prediction failed"
}


def find_free_port() -> int:
    """
    Find an available port on the host system.

    Returns
    -------
    int
        An available port number.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_docker_container(image_name: str, internal_port: int = 8000) -> Tuple[docker.models.containers.Container, int]:
    """
    Pull the Docker image (if needed) and start the container on a random free port.

    Parameters
    ----------
    image_name : str
        Name of the Docker image.
    internal_port : int, optional
        The container's exposed port, by default 8000.

    Returns
    -------
    Tuple[docker.models.containers.Container, int]
        The container instance and the bound host port.
    """
    client = docker.from_env()

    try:
        client.images.pull(image_name)
        print(f"[DEBUG] Successfully pulled image: {image_name}")
    except Exception as exc:
        print(f"[WARN] Could not pull image {image_name}, continuing locally: {exc}")

    host_port = find_free_port()
    print(f"[DEBUG] Launching container on host port {host_port}...")

    container = client.containers.run(
        image=image_name,
        detach=True,
        remove=True,
        ports={f"{internal_port}/tcp": host_port}
    )
    time.sleep(1)

    container.reload()
    print(f"[DEBUG] Container status: {container.status}")
    if container.status != "running":
        logs = container.logs().decode(errors="ignore")
        print(f"[ERROR] Container not in 'running' state. Logs:\n{logs}")

    return container, host_port


def wait_for_container(host_port: int, timeout: int = 10) -> None:
    """
    Wait for the container to respond on the given host port.

    Parameters
    ----------
    host_port : int
        The bound host port.
    timeout : int, optional
        Maximum wait time in seconds, by default 10.

    Raises
    ------
    RuntimeError
        If the container doesn't respond within the timeout.
    """
    start = time.time()
    while (time.time() - start) < timeout:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(("localhost", host_port)) == 0:
                print(f"[DEBUG] Container is responding on port {host_port}")
                return
        time.sleep(0.5)
    raise RuntimeError("Docker container did not become ready within timeout.")


def request_prediction(base_url: str, payload: Dict[str, Any]) -> None:
    """
    Send a POST request to /predict with the input payload.

    Parameters
    ----------
    base_url : str
        Base URL of the container, e.g., 'http://localhost:12345'
    payload : Dict[str, Any]
        JSON payload to send to /predict endpoint.

    Raises
    ------
    requests.exceptions.HTTPError
        If the response status is not successful.
    """
    print(f"[DEBUG] Sending payload to {base_url}/predict: {payload}")
    resp = requests.post(f"{base_url}/predict", json=payload)
    resp.raise_for_status()


def get_status_code(base_url: str) -> int:
    """
    Retrieve the current model status code from /status.

    The endpoint is expected to return JSON of the form:
      {"status": <int>, "message": <str>}
    """
    resp = requests.get(f"{base_url}/status")
    if resp.ok:
        try:
            data = resp.json()
            code = data.get("status", -1)
            msg = data.get("message", "")
            print(f"[DEBUG] Status code: {code}, message: {msg}")
            return code
        except Exception as ex:
            print(f"[WARN] Could not parse JSON from /status: {ex}")
    else:
        print(f"[WARN] /status request failed with code {resp.status_code}.")
    return -1


def retrieve_result(base_url: str) -> list[float]:
    """
    Get numeric result(s) from /result as a list.

    The endpoint may return either a single numeric value or a list of numerics.

    Returns
    -------
    list[float]
        The numeric model result(s) as a list of floats.

    Raises
    ------
    RuntimeError
        If result cannot be parsed.
    """
    resp = requests.get(f"{base_url}/result")
    resp.raise_for_status()

    try:
        data = resp.json()  # Attempt to parse JSON.
        if isinstance(data, list):
            # e.g. [0.2, 0.5, 0.7]
            return [float(x) for x in data]
        else:
            # Single numeric (e.g. 0.1234 or "0.1234")
            return [float(data)]
    except Exception as ex:
        raise RuntimeError(f"Failed to parse result from /result: {ex}")
        


def stop_docker_container(container: docker.models.containers.Container) -> None:
    """
    Stop the Docker container.

    Parameters
    ----------
    container : docker.models.containers.Container
        The running container instance.
    """
    print("[DEBUG] Stopping container...")
    container.stop()


def execute_model(metadata: Any, input_payload: Dict[str, Any]) -> list[float]:
    """
    Multi-step model execution:
      1) Start container.
      2) Wait for readiness.
      3) POST /predict.
      4) Poll /status for code == 3 (completed) or 4 (failed).
      5) If completed, GET /result.
      6) Stop container.

    Parameters
    ----------
    metadata : Any
        Metadata with 'docker_image' key or attribute.
    input_payload : Dict[str, Any]
        Inputs for the model.

    Returns
    -------
    list[float]
        Numeric prediction result(s).

    Raises
    ------
    RuntimeError
        If container fails or if it times out waiting for completion.
    """
    container, port = start_docker_container(metadata.docker_image)
    base_url = f"http://localhost:{port}"

    try:
        # 1) Wait for container
        wait_for_container(port)

        # 2) Request prediction
        request_prediction(base_url, input_payload)

        # 3) Poll /status
        start_time = time.time()
        timeout = 60
        while time.time() - start_time < timeout:
            code = get_status_code(base_url)
            if code == 3:  # 'Prediction completed'
                val = retrieve_result(base_url)
                print(f"[DEBUG] Final numeric result: {val}")
                return val
            elif code == 4:  # 'Prediction failed'
                raise RuntimeError("Model returned a failed status.")

            time.sleep(1)

        raise RuntimeError("Timed out waiting for model to complete.")

    finally:
        stop_docker_container(container)
