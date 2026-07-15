from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class BackBoneIdentity(nn.Module):
    """
    Dummy backbone for ablations.

    It forwards latent sequences unchanged from the visual encoder to the visual
    decoder, while optionally validating the expected sequence length.
    """

    def __init__(
        self,
        sequence_length: int | None = None,
        transformer: object | None = None,
    ) -> None:
        super().__init__()
        if sequence_length is not None and int(sequence_length) <= 0:
            raise ValueError("sequence_length must be positive when provided.")
        self.sequence_length = None if sequence_length is None else int(sequence_length)
        # Kept only so the shared train hack can write transformer-shaped dummy fields.
        self.transformer = transformer

    def forward(
        self,
        latents: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents.dim() != 3:
            raise ValueError("latents must have shape (batch, seq_len, embed_dim).")
        if self.sequence_length is not None and latents.size(1) != self.sequence_length:
            raise ValueError(
                f"Expected latent sequence length {self.sequence_length}, received {latents.size(1)}."
            )
        if padding_mask is not None and padding_mask.shape != latents.shape[:2]:
            raise ValueError("padding_mask must have shape (batch, seq_len).")
        return latents
