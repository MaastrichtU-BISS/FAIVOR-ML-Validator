import docker
import requests
import socket
import time
from contextlib import closing
from typing import Dict, Any, Tuple


def find_free_port() -> int:
    """
    Find an available port on the host system to bind.

    Returns
    -------
    int
        An available port number.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_docker_container(image_name: str, internal_port: int = 8000) -> Tuple[docker.models.containers.Container, int]:
    """
    Run a Docker container from the specified image on a free port.

    Parameters
    ----------
    image_name : str
        Name of the Docker image to run.
    internal_port : int, optional
        The internal port the container is listening on (defaults to 8000).

    Returns
    -------
    Tuple[docker.models.containers.Container, int]
        A 2-tuple: (container object, allocated host port).
    """
    client = docker.from_env()

    # Try pulling the image (no-op if already present)
    try:
        client.images.pull(image_name)
    except Exception as exc:
        print(f"Warning: Could not pull image {image_name}, continuing: {exc}")

    port = find_free_port()
    print(f"[DEBUG] Using free port on host: {port}")

    container = client.containers.run(
        image=image_name,
        detach=True,
        remove=True,
        ports={f"{internal_port}/tcp": port}
    )

    # Give Docker a moment to register container start
    time.sleep(1)

    container.reload()
    print(f"[DEBUG] Container status after run: {container.status}")

    # If container is not running, log output for debugging
    if container.status != "running":
        logs = container.logs().decode(errors="ignore")
        print(f"[ERROR] Container failed to start. Logs:\n{logs}")

    return container, port


def wait_for_container(port: int, timeout: int = 10) -> None:
    """
    Wait until the Docker container is ready to accept connections.

    Parameters
    ----------
    port : int
        The port on which the container is expected to respond.
    timeout : int
        Maximum number of seconds to wait for the container.

    Raises
    ------
    RuntimeError
        If the container does not become responsive within the timeout.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(("localhost", port)) == 0:
                print("[DEBUG] Container is responding on port", port)
                return
        time.sleep(0.5)
    raise RuntimeError("Docker container did not start in time.")


def execute_model(metadata: Any, input_payload: Dict[str, Any]) -> float:
    """
    Multi-step execution of a model that uses status polling:
      1) POST to /predict
      2) Poll /status until 'Prediction completed' or 'Prediction failed'
      3) If completed, GET /result for numeric outcome.

    Parameters
    ----------
    metadata : Any
        Metadata object containing Docker image information.
    input_payload : Dict[str, Any]
        JSON payload with input data to send to the model.

    Returns
    -------
    float
        The numeric result retrieved from /result.

    Raises
    ------
    RuntimeError
        If the container fails to become responsive, model fails, or no result is returned.
    """
    container, port = run_docker_container(metadata.docker_image)

    try:
        # 1) Wait for container readiness
        wait_for_container(port)

        base_url = f"http://localhost:{port}"
        print(f"[DEBUG] Sending POST to {base_url}/predict")
        resp_pred = requests.post(f"{base_url}/predict", json=input_payload)
        resp_pred.raise_for_status()

        # 2) Poll /status for completion
        start_time = time.time()
        timeout = 60  # seconds to wait for model completion
        status_text = ""
        while time.time() - start_time < timeout:
            resp_status = requests.get(f"{base_url}/status")
            if resp_status.ok:
                status_text = resp_status.text.strip()
                print(f"[DEBUG] Current status: {status_text}")
                if status_text == "Prediction completed":
                    break
                elif status_text == "Prediction failed":
                    raise RuntimeError("Model returned failed status.")
            time.sleep(1)
        else:
            raise RuntimeError("Timed out waiting for model to complete.")

        # 3) Retrieve numeric result from /result
        resp_result = requests.get(f"{base_url}/result")
        resp_result.raise_for_status()
        try:
            # if JSON parse fails, fallback to text
            result_val = float(resp_result.json())
        except Exception:
            result_val = float(resp_result.text)

        print("[DEBUG] Final numeric result:", result_val)
        return result_val

    finally:
        print("[DEBUG] Stopping container.")
        container.stop()
