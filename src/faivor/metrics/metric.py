from sklearn import metrics as skm
from typing import Callable, List, Optional
import numpy as np
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError


class ModelMetric:
    """Class to encapsulate any metric."""
    def __init__(
        self,
        function_name: str,
        regular_name: str,
        description: str,
        func: Callable,
        is_torch: bool = False,  
        torch_kwargs: Optional[dict] = None,
    ):
        self.function_name = function_name # Name of the function
        self.regular_name = regular_name  # Human-readable name
        self.description = description  # Description of the metric
        self.func = func  # Actual function
        self.is_torch = is_torch  # Flag to check if the metric uses torch
        self.torch_kwargs = torch_kwargs or {}

    def compute(self, y_true, y_pred, **kwargs):
        """Compute the metric."""
        if self.is_torch:
            metric = self.func(**self.torch_kwargs)
            return metric(
                torch.tensor(y_pred, dtype=torch.float32),
                torch.tensor(y_true, dtype=torch.float32),
            ).detach().cpu().item()
        return self.func(y_true, y_pred, **kwargs)