from __future__ import annotations

import torch
from torch import nn


class GLoOD(nn.Module):
    """
    PyTorch refactor of the GLoOD model with an internal ``nn.Transformer`` backbone.

    Parameters
    ----------
    visual_model:
        Autoencoder providing ``encode`` and ``decode`` methods that operate on
        Autoencoder providing ``encode`` and ``decode`` methods that operate on
        tensors with shape ``(batch, height, width, channels)``.
    backbone:
        Transformer backbone that applies sinusoidal positional encodings and masking
        to the latent sequence prior to decoding.
    sequence_length:
        Number of latent frames expected per example (typically ``len_seq - 1`` from the dataset config).
    """

    def __init__(
        self,
        visual_model: nn.Module,
        backbone: nn.Module,
        n_seq: int,
        h: int,
        w: int,
        c: int,
    ) -> None:
        super().__init__()
        self.visual_model = visual_model
        if not isinstance(backbone, nn.Module):
            raise TypeError("GLoOD backbone must be an nn.Module instance.")
        self.backbone = backbone

        for name, value in (("n_seq", n_seq), ("h", h), ("w", w), ("c", c)):
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        self.n_seq = int(n_seq)
        self.h = int(h)
        self.w = int(w)
        self.c = int(c)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 5:
            raise ValueError("GLoOD expects inputs with shape (batch, n_seq, height, width, channels).")
        if (
            inputs.size(1) != self.n_seq
            or inputs.size(2) != self.h
            or inputs.size(3) != self.w
            or inputs.size(4) != self.c
        ):
            raise ValueError(
                f"Expected input shape (batch, {self.n_seq}, {self.h}, {self.w}, {self.c});"
                f" received {tuple(inputs.shape)}."
            )

        latents, skips = self._apply_visual_model_encode(inputs)
        padding_mask = torch.all(latents == 0, dim=-1)
        transformed_latents = self.backbone(latents, padding_mask)
        decoded = self._apply_visual_model_decode(transformed_latents, skips)
        return decoded

    def _apply_visual_model_encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size * self.n_seq, self.h, self.w, self.c)

        latents_flat, skips_flat = self.visual_model.encode(x_flat)
        if latents_flat.dim() != 2:
            raise ValueError("visual_model.encode must return latents with shape (batch, latent_dim).")
        latent_dim = latents_flat.size(-1)

        latents = latents_flat.view(batch_size, self.n_seq, latent_dim)
        return latents, skips_flat

    def _apply_visual_model_decode(self, latents: torch.Tensor, skips: torch.Tensor) -> torch.Tensor:
        batch_size = latents.size(0)
        latent_dim = latents.size(-1)
        z_flat = latents.reshape(batch_size * self.n_seq, latent_dim)

        recon_flat = self.visual_model.decode(z_flat, skips)
        if recon_flat.size(-1) != self.c:
            raise ValueError("visual_model.decode must return tensors with the last dimension equal to 3 (channels).")

        recon = recon_flat.view(batch_size, self.n_seq, self.h, self.w, self.c)
        return recon
