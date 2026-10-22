import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from faivor.Calibration import Calibration
from faivor.api_controller import app
from faivor.model_metadata import ModelMetadata

client = TestClient(app)

OUTCOME = "outcome"
SUMMARY_KEYS = {
    "predictions",
    "log_likelihoods",
    "model_selection",
    "roc_metrics",
    "calibration_curve",
    "coefficients",
}


def make_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    true_prob = 1 / (1 + np.exp(-(-0.5 + 0.8 * x1 + 0.4 * x2)))
    y = rng.binomial(1, true_prob)
    predictions = np.clip(true_prob + rng.normal(scale=0.05, size=n), 0.01, 0.99).tolist()
    expected_outputs = [{OUTCOME: int(v)} for v in y]
    features = pd.DataFrame({"x1": x1, "x2": x2})
    return predictions, expected_outputs, features


@pytest.fixture
def calibration() -> Calibration:
    predictions, expected_outputs, features = make_data()
    return Calibration.from_docker_output(
        docker_result={"predictions": predictions},
        expected_outputs=expected_outputs,
        outcome_col=OUTCOME,
        input_data=features,
    )


def test_from_docker_output_happy_path(calibration: Calibration):
    assert len(calibration.data) == 200
    assert {"predictions", OUTCOME, "x1", "x2"} <= set(calibration.data.columns)


def test_get_calibration_summary_keys(calibration: Calibration):
    summary = calibration.get_calibration_summary()
    assert SUMMARY_KEYS <= set(summary)
    assert summary["model_selection"]["selected_model"] in {"m0", "m1", "m2", "m3"}
    assert set(summary["roc_metrics"]) == {"m0", "m1", "m2", "m3"}
    for metrics in summary["roc_metrics"].values():
        assert set(metrics) == {"fpr", "tpr", "thresholds", "auc"}
        assert all(isinstance(metrics[key], list) for key in ("fpr", "tpr", "thresholds"))
    assert set(summary["calibration_curve"]) == {"m0", "m1", "m2", "m3"}
    for curve in summary["calibration_curve"].values():
        assert set(curve) == {"observed_probability", "predicted_probability"}
        assert isinstance(curve["observed_probability"], list)
        assert isinstance(curve["predicted_probability"], list)
    assert set(summary["coefficients"]) == {"m1", "m2", "m3"}


def test_get_model_coefficients(calibration: Calibration):
    coefficients = calibration.get_model_coefficients()

    assert set(coefficients) == {"m1", "m2", "m3"}
    assert set(coefficients["m1"]) == {"intercept"}
    assert set(coefficients["m2"]) == {"intercept", "slope"}
    assert set(coefficients["m3"]) == {"intercept", "covariates"}
    assert set(coefficients["m3"]["covariates"]) == {"x1", "x2"}
    assert all(
        isinstance(value, float)
        for model in (coefficients["m1"], coefficients["m2"])
        for value in model.values()
    )
    assert isinstance(coefficients["m3"]["intercept"], float)
    assert all(isinstance(value, float) for value in coefficients["m3"]["covariates"].values())
    assert coefficients["m1"]["intercept"] == pytest.approx(
        calibration._fitted_m1.params["Intercept"]
    )
    assert coefficients["m2"]["slope"] == pytest.approx(
        calibration._fitted_m2.params["linear_predictor"]
    )
    assert coefficients["m3"]["covariates"]["x1"] == pytest.approx(
        calibration._fitted_m3.params["x1"]
    )


@pytest.mark.parametrize("covariate_columns", [["x1", "x2"], None])
def test_get_results_table(calibration: Calibration, covariate_columns):
    table = calibration.get_results_table(covariate_columns=covariate_columns)
    variables = list(table["Variable"])
    assert {"x1", "x2", "Intercept", "AUC"} <= set(variables)


def test_empty_predictions_raises():
    _, expected_outputs, _ = make_data(5)
    with pytest.raises(ValueError):
        Calibration(predictions=[], expected_outputs=expected_outputs, outcome_col=OUTCOME)


def test_mismatched_lengths_raises():
    predictions, expected_outputs, _ = make_data(10)
    with pytest.raises(ValueError):
        Calibration(predictions=predictions[:5], expected_outputs=expected_outputs, outcome_col=OUTCOME)


def test_from_docker_output_missing_predictions_raises():
    _, expected_outputs, _ = make_data(5)
    with pytest.raises(ValueError):
        Calibration.from_docker_output({}, expected_outputs, OUTCOME)


def _load_model(shared_datadir: Path, model_name: str = "pilot-model_1"):
    model_dir = shared_datadir / "models" / model_name
    with open(model_dir / "metadata.json", encoding="utf-8") as f:
        metadata_str = f.read()
    return metadata_str, model_dir / "data.csv", model_dir / "column_metadata.json"


def _fake_execute_model(metadata, inputs):
    rng = np.random.default_rng(1)
    return {"predictions": rng.uniform(0.05, 0.95, size=len(inputs)).tolist()}


def test_calibrate_model_endpoint(shared_datadir: Path):
    metadata_str, csv_path, column_metadata_path = _load_model(shared_datadir)
    with open(csv_path, "rb") as csv_file, patch(
        "faivor.api_controller.execute_model", side_effect=_fake_execute_model
    ):
        files = {
            "model_metadata": (None, metadata_str),
            "csv_file": ("data.csv", csv_file, "text/csv"),
            "column_metadata": (None, column_metadata_path.read_text(encoding="utf-8")),
        }
        response = client.post("/calibrate-model", files=files)

    assert response.status_code == 200, response.text
    assert SUMMARY_KEYS <= set(response.json())


def test_calibrate_model_invalid_metadata_json(shared_datadir: Path):
    _, csv_path, _ = _load_model(shared_datadir)
    with open(csv_path, "rb") as csv_file:
        files = {
            "model_metadata": (None, "{not valid json"),
            "csv_file": ("data.csv", csv_file, "text/csv"),
        }
        response = client.post("/calibrate-model", files=files)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_METADATA_JSON"
