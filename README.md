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

**Important:** This service requires Docker access to validate ML models. When running in a container, you must mount the Docker socket. The service automatically handles Docker-in-Docker networking by detecting the environment and using the appropriate hostname. See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.

# How to Implement New Metrics in FAIVOR-ML-Validator

This guide explains how to add new performance, fairness, or explainability metrics to the FAIVOR-ML-Validator framework. You can add metrics based on scikit-learn, PyTorch, or your own custom Python functions.

---

## 1. Decide the Metric Type and Location

- **Classification metrics:**  
  Place code in `performance.py`, `fairness.py`, or `explainability.py`.
- **Regression metrics:**  
  Place code in `performance.py`, `fairness.py`, or `explainability.py`.

---

## 2. Implement the Metric Function

- **For scikit-learn metrics:**  
  You can reference them directly in the YAML config (see below).
- **For custom metrics:**  
  Write a Python function with the signature:
  ```python
  def my_metric(y_true, y_pred, **kwargs):
      # ... your logic ...
      return result
  ```
  Example (classification):
  ```python
  # filepath: src/faivor/metrics/classification/performance.py
  def my_custom_metric(y_true, y_pred):
      # Example: return the mean absolute error
      return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
  ```

---

## 3. Register the Metric in the YAML Config

- **Classification:**  
  Edit classification_metrics.yaml
- **Regression:**  
  Edit regression_metrics.yaml

Add a new entry under the appropriate section (`performance`, `fairness`, or `explainability`):

```yaml
- function_name: my_custom_metric
  regular_name: My Custom Metric
  description: Computes the mean absolute error as a custom metric.
  func: faivor.metrics.classification.performance.my_custom_metric
  is_torch: false
```

- `function_name`: The Python function name.
- `regular_name`: Human-readable name.
- `description`: Short description.
- `func`: Full import path to your function (or scikit-learn function).
- `is_torch`: Set to `true` if your function uses PyTorch tensors.

---

## 4. (Optional) Add Unit Tests

Add tests for your metric in the appropriate test file, e.g.:
- test_performance.py
- test_performance.py

Example:
```python
def test_my_custom_metric():
    y_true = [1, 0, 1]
    y_pred = [1, 1, 0]
    assert my_custom_metric(y_true, y_pred) == pytest.approx(0.666, rel=1e-2)
```

---

## 5. Use the Metric

Your new metric will now be available for automatic calculation and reporting in the FAIVOR-ML-Validator pipeline, including subgroup and threshold analyses.

---

## 6. Troubleshooting

- **Import errors:**  
  Ensure your function is imported in the module’s __init__.py if needed.
- **YAML path errors:**  
  The `func` path must be correct and importable from the project root.
- **Signature mismatch:**  
  Your function should accept at least `y_true` and `y_pred` as arguments.

---

## Example: Adding a Custom Regression Metric

1. **Implement:**
   ```python
   # filepath: src/faivor/metrics/regression/performance.py
   def mean_absolute_percentage_error(y_true, y_pred):
       y_true, y_pred = np.array(y_true), np.array(y_pred)
       return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
   ```
2. **Register:**
   ```yaml
   # filepath: src/faivor/metrics/regression/regression_metrics.yaml
   - function_name: mean_absolute_percentage_error
     regular_name: Mean Absolute Percentage Error
     description: Computes the mean absolute percentage error.
     func: faivor.metrics.regression.performance.mean_absolute_percentage_error
     is_torch: false
   ```
3. **Test (optional):**
   ```python
   def test_mean_absolute_percentage_error():
       y_true = [100, 200, 300]
       y_pred = [110, 190, 310]
       assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(4.44, rel=1e-2)
   ```

---

## 7. Advanced: Metrics Requiring Extra Arguments

If your metric needs extra arguments (e.g., sensitive attributes or feature importances), add them as keyword arguments. The framework will pass them if available.

Example:
```python
def fairness_metric(y_true, y_pred, sensitive_values=None):
    # ... logic using sensitive_values ...
```

---

## 8. Reload and Validate

After adding your metric, restart the service or rerun your scripts to ensure the new metric is loaded and available.

---

**For more details, see the existing metric implementations in the metrics directory.**

## Local development

To get started with this project, please follow these steps to set up the environment and configure Jupyter for notebook use:

