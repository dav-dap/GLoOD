from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class BackBoneTransformerSinusoidal(nn.Module):
    """
    Transformer backbone with non-trainable sinusoidal positional encodings and attention masking.

    Parameters
    ----------
    transformer:
        Instantiated :class:`torch.nn.Transformer` whose ``d_model`` matches the latent embedding size.
    sequence_length:
        Number of temporal positions expected by the transformer (latent sequence length).
    """

    def __init__(self, transformer: nn.Transformer, sequence_length: int) -> None:
        super().__init__()
        self.transformer = transformer
        if sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer.")
        self.sequence_length = int(sequence_length)

        embed_dim = getattr(self.transformer, "d_model", None)
        if embed_dim is None:
            raise ValueError("Transformer must expose a d_model attribute.")

        positional_encoding = self._build_positional_encoding(self.sequence_length, embed_dim, torch.float32)
        look_ahead_mask = self._generate_look_ahead_mask(self.sequence_length)

        self.register_buffer("_positional_encoding", positional_encoding, persistent=False)
        self.register_buffer("_look_ahead_mask", look_ahead_mask, persistent=False)

        encoder = getattr(self.transformer, "encoder", None)
        if encoder is not None and hasattr(encoder, "use_nested_tensor"):
            encoder.enable_nested_tensor = False
            encoder.use_nested_tensor = False

    def forward(
        self,
        latents: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        latents:
            Tensor with shape ``(batch, seq_len, embed_dim)`` representing encoded frames.
        padding_mask:
            Optional boolean tensor with shape ``(batch, seq_len)`` where ``True`` marks padded tokens.
        """
        if latents.dim() != 3:
            raise ValueError("latents must have shape (batch, seq_len, embed_dim).")

        batch_size, seq_len, embed_dim = latents.shape
        if seq_len != self.sequence_length:
            raise ValueError(
                f"Expected latent sequence length {self.sequence_length}, received {seq_len}."
            )

        transformer_embed_dim = getattr(self.transformer, "d_model", None)
        if transformer_embed_dim is not None and transformer_embed_dim != embed_dim:
            raise ValueError(
                f"Transformer d_model={transformer_embed_dim} does not match latent dimension {embed_dim}."
            )

        positional_encoding = self._positional_encoding.to(device=latents.device, dtype=latents.dtype)
        encoder_input = latents + positional_encoding
        decoder_input = encoder_input

        attn_padding_mask = None
        if padding_mask is not None:
            if padding_mask.shape != (batch_size, seq_len):
                raise ValueError("padding_mask must have shape (batch, seq_len).")
            attn_padding_mask = padding_mask.to(dtype=torch.bool, device=latents.device)

        tgt_mask = self._look_ahead_mask.to(device=latents.device)
        return self.transformer(
            encoder_input,
            decoder_input,
            tgt_mask=tgt_mask,
            src_key_padding_mask=attn_padding_mask,
            tgt_key_padding_mask=attn_padding_mask,
            memory_key_padding_mask=attn_padding_mask,
        )

    @staticmethod
    def _build_positional_encoding(
        seq_len: int,
        embed_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.arange(seq_len, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=dtype) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(seq_len, embed_dim, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    @staticmethod
    def _generate_look_ahead_mask(
        seq_len: int,
    ) -> torch.Tensor:
        if seq_len <= 0:
            raise ValueError("Sequence length must be positive to generate a look-ahead mask.")
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        return mask
