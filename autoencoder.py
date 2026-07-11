"""Convolutional latent autoencoder for mel spectrograms."""

import torch
import torch.nn as nn


class LatentEncoder(nn.Module):
    """
    Convolutional encoder: mel spectrogram -> 2D latent feature map.
    (1, 128, T) -> (C_lat, 8, T/16) via 4x stride-2 downsamples.
    """

    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = [1, 32, 64, 128, 32]
        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], 4, stride=2, padding=1),
                nn.GroupNorm(min(8, channels[i + 1]), channels[i + 1]),
                nn.SiLU(),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentDecoder(nn.Module):
    """
    Convolutional decoder: 2D latent feature map -> mel spectrogram.
    (C_lat, 8, T/16) -> (1, 128, T) via 4x stride-2 upsamples.
    """

    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = [32, 128, 64, 32, 1]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                nn.ConvTranspose2d(channels[i], channels[i + 1], 4,
                                   stride=2, padding=1)
            )
            if i < len(channels) - 2:
                layers.extend([
                    nn.GroupNorm(min(8, channels[i + 1]), channels[i + 1]),
                    nn.SiLU(),
                ])
            else:
                layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentAutoencoder(nn.Module):
    """Convolutional autoencoder for mel spectrograms (MSE loss, no KL)."""

    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = [1, 32, 64, 128, 32]
        self.encoder = LatentEncoder(channels)
        self.decoder = LatentDecoder(channels[::-1])

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
