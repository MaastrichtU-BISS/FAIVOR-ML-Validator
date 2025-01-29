from dataclasses import dataclass, field
from typing import Callable, Optional, Dict
import torch

@dataclass
class ModelMetric:
    function_name: str
    regular_name: str
    description: str
    func: Callable
    is_torch: bool = False
    torch_kwargs: Dict = field(default_factory=dict)

    def compute(self, y_true, y_pred, **kwargs) -> float:
        """Compute the metric based on whether it's a Torch or Sklearn function."""
        if self.is_torch:
            metric = self.func(**self.torch_kwargs)
            return metric(
                torch.tensor(y_pred, dtype=torch.float32),
                torch.tensor(y_true, dtype=torch.float32),
            ).detach().cpu().item()
        return self.func(y_true, y_pred, **kwargs)