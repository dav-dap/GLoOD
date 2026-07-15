from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class UNet_tri_decoupled(nn.Module):
    """
    PyTorch port of the GLoOD tri-decoupled U-Net autoencoder.
    Expects inputs with shape (batch, 101, 101, 3) and returns tensors
    of the same shape.
    """

    def __init__(self, single_latent_dim: int = 256) -> None:
        super().__init__()
        self.single_latent_dim = single_latent_dim

        # Shapes refer to (channels, height, width) for channels-first tensors.
        self.skip_shapes = [(8, 52, 52), (16, 26, 26), (32, 13, 13)]
        self.skip_sizes = [c * h * w for c, h, w in self.skip_shapes]

        self.encoder_p = self._build_encoder()
        self.encoder_ux = self._build_encoder()
        self.encoder_uy = self._build_encoder()

        self.decoder_p = self._build_decoder()
        self.decoder_ux = self._build_decoder()
        self.decoder_uy = self._build_decoder()

        in_latent = 3 * self.single_latent_dim
        self.dense_encoder = nn.Linear(in_latent, in_latent)
        self.dense_decoder = nn.Linear(in_latent, in_latent)

        self.act = nn.ELU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _build_encoder(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "encoder_up": nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                "encoder_me": nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                "encoder_lo": nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                "bottleneck_dense": nn.Linear(32 * 13 * 13, self.single_latent_dim),
            }
        )

    def _build_decoder(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "bottleneck_dense": nn.Linear(self.single_latent_dim, 32 * 13 * 13),
                "decoder_lo": nn.ConvTranspose2d(
                    32 + 32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                "decoder_me": nn.ConvTranspose2d(
                    16 + 16, 8, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                "decoder_up": nn.ConvTranspose2d(
                    8 + 8, 1, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
            }
        )

    def _encode_field(
        self, x: torch.Tensor, encoder: nn.ModuleDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        x_up = self.act(encoder["encoder_up"](x))
        x_me = self.act(encoder["encoder_me"](x_up))
        x_lo = self.act(encoder["encoder_lo"](x_me))

        latent = torch.flatten(x_lo, start_dim=1)
        latent = self.act(encoder["bottleneck_dense"](latent))

        skips = torch.cat([tensor.flatten(start_dim=1) for tensor in (x_up, x_me, x_lo)], dim=-1)
        return latent, skips

    def _decode_field(
        self,
        latent_x: torch.Tensor,
        skips: torch.Tensor,
        decoder: nn.ModuleDict,
    ) -> torch.Tensor:
        batch_size = latent_x.shape[0]

        skip_up, skip_me, skip_lo = torch.split(skips, self.skip_sizes, dim=-1)
        skip_up = skip_up.view(batch_size, *self.skip_shapes[0])
        skip_me = skip_me.view(batch_size, *self.skip_shapes[1])
        skip_lo = skip_lo.view(batch_size, *self.skip_shapes[2])

        x = self.act(decoder["bottleneck_dense"](latent_x))
        x = x.view(batch_size, 32, 13, 13)

        x = torch.cat([x, skip_lo], dim=1)
        x = self.act(decoder["decoder_lo"](x))

        x = torch.cat([x, skip_me], dim=1)
        x = self.act(decoder["decoder_me"](x))

        x = torch.cat([x, skip_up], dim=1)
        x = self.act(decoder["decoder_up"](x))

        return x

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.dim() != 4:
            raise ValueError("Expected inputs with shape (batch, height, width, channels).")
        if inputs.size(-1) != 3:
            raise ValueError("Expected inputs with three channels.")

        x = inputs.permute(0, 3, 1, 2)
        x = F.pad(x, (0, 3, 0, 3))

        p, ux, uy = torch.chunk(x, chunks=3, dim=1)
        latent_p, skips_p = self._encode_field(p, self.encoder_p)
        latent_ux, skips_ux = self._encode_field(ux, self.encoder_ux)
        latent_uy, skips_uy = self._encode_field(uy, self.encoder_uy)

        latent = torch.cat([latent_p, latent_ux, latent_uy], dim=-1)
        skips = torch.cat([skips_p, skips_ux, skips_uy], dim=-1)

        latent = self.act(self.dense_encoder(latent))
        return latent, skips

    def decode(
        self, latent: torch.Tensor, skips: torch.Tensor
    ) -> torch.Tensor:
        batch_size = latent.shape[0]

        latent = self.act(self.dense_decoder(latent))
        latent_p, latent_ux, latent_uy = torch.chunk(latent, chunks=3, dim=-1)
        skips_p, skips_ux, skips_uy = torch.chunk(skips, chunks=3, dim=-1)

        p = self._decode_field(latent_p, skips_p, self.decoder_p)
        ux = self._decode_field(latent_ux, skips_ux, self.decoder_ux)
        uy = self._decode_field(latent_uy, skips_uy, self.decoder_uy)

        outputs = torch.cat([p, ux, uy], dim=1)
        outputs = outputs[:, :, :101, :101]
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent, skips = self.encode(inputs)
        return self.decode(latent, skips)
