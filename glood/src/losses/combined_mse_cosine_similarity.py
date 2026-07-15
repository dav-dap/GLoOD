from __future__ import annotations

import torch
from torch import nn


class CombinedMseSimilarityLoss(nn.Module):
    """
    PyTorch port of the FluidGPT combined MSE + cosine-similarity loss.

    The loss blends the full-tensor MSE with a cosine-distance component over
    the velocity channels (components 1 and 2), mirroring the TensorFlow version.
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in the [0, 1] interval.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = torch.mean((preds - targets) ** 2)

        pred_velocity = preds[..., 1:3]
        true_velocity = targets[..., 1:3]

        dot_product = torch.sum(pred_velocity * true_velocity, dim=-1)
        magnitude_pred = torch.norm(pred_velocity, dim=-1)
        magnitude_true = torch.norm(true_velocity, dim=-1)
        # Clamp the magnitudes product to keep the cosine term numerically stable.
        denom = torch.clamp(magnitude_pred * magnitude_true, min=self.eps)
        cosine = dot_product / denom
        cosine_loss = 1.0 - torch.mean(cosine)

        return self.alpha * mse_loss + (1.0 - self.alpha) * cosine_loss
