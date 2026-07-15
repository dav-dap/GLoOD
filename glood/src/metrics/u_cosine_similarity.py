from __future__ import annotations

import torch


def u_cosine_similarity(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine-distance style metric over the velocity channels (components 1 and 2).
    Returns 1 - mean(cosine_similarity) to match the TensorFlow implementation.
    """
    pred_velocity = preds[..., 1:3]
    true_velocity = targets[..., 1:3]

    dot_product = torch.sum(pred_velocity * true_velocity, dim=-1)
    magnitude_pred = torch.norm(pred_velocity, dim=-1)
    magnitude_true = torch.norm(true_velocity, dim=-1)

    denom = torch.clamp(magnitude_pred * magnitude_true, min=eps)
    cosine = dot_product / denom
    cosine = torch.clamp(cosine, -1.0, 1.0)

    return 1.0 - torch.mean(cosine)
