from __future__ import annotations

import torch


def u_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error across the velocity channels (components 1 and 2).
    """
    pred_velocity = preds[..., 1:3]
    true_velocity = targets[..., 1:3]
    return torch.mean((pred_velocity - true_velocity) ** 2)
