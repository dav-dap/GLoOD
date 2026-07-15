from __future__ import annotations

import torch


def p_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error on the pressure channel (component 0).
    """
    pred_pressure = preds[..., 0]
    true_pressure = targets[..., 0]
    return torch.mean((pred_pressure - true_pressure) ** 2)
