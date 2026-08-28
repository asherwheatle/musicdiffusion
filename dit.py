"""Diffusion Transformer (DiT) + ControlNet (paper sections III-A, III-B)."""

import math
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Norm: scale and shift conditioned on timestep embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.linear = nn.Linear(d_model, d_model * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.linear(t_emb).unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    """
    Transformer block: AdaLN -> SelfAttn -> AdaLN -> CrossAttn(text) -> AdaLN -> MLP

    Timestep conditioning via AdaLN, text conditioning via cross-attention.
    The ControlNet branch clones these blocks (paper section III-A).
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.adaln_sa = AdaLayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

        self.adaln_ca = AdaLayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

        self.adaln_mlp = AdaLayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                text_emb: torch.Tensor) -> torch.Tensor:
        h = self.adaln_sa(x, t_emb)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        h = self.adaln_ca(x, t_emb)
        h, _ = self.cross_attn(h, text_emb, text_emb)
        x = x + h

        h = self.adaln_mlp(x, t_emb)
        x = x + self.mlp(h)
        return x


class ZeroLinear(nn.Module):
    """Zero-initialized linear layer for ControlNet (paper section III-A)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MoodDiT(nn.Module):
    """
    Full Diffusion Transformer with ControlNet branch.

    Paper architecture (section III-A):
      - N DiT blocks conditioned on text (cross-attn) and timestep (AdaLN)
      - First N/2 blocks have parallel ControlNet branches receiving melody
      - ControlNet outputs pass through ZeroLinear before adding to DiT stream
        so melody conditioning starts from zero and ramps up smoothly
    """

    def __init__(self, latent_channels: int = 32, latent_h: int = 8,
                 d_model: int = 256, n_heads: int = 4,
                 n_blocks: int = 8, n_control_blocks: int = 4,
                 mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.d_model = d_model

        self.input_proj = nn.Linear(latent_channels, d_model)
        self.output_proj = nn.Linear(d_model, latent_channels)

        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed = SinusoidalTimestepEmbedding(d_model)

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])

        self.n_control_blocks = n_control_blocks
        self.control_blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_control_blocks)
        ])
        self.zero_linears = nn.ModuleList([
            ZeroLinear(d_model) for _ in range(n_control_blocks)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, z_noisy: torch.Tensor, timesteps: torch.Tensor,
                text_emb: torch.Tensor, melody_emb: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            z_noisy:    (B, C_lat, H_lat, W_lat) noisy latent
            timesteps:  (B,) diffusion timestep indices
            text_emb:   (B, text_seq, d_model) from ClapTextEncoder
            melody_emb: (B, W_lat, d_model) from MelodyEncoder
        Returns:
            (B, C_lat, H_lat, W_lat) predicted v (velocity)
        """
        B, C, H, W = z_noisy.shape
        seq_len = H * W

        x = z_noisy.reshape(B, C, seq_len).permute(0, 2, 1)  # (B, seq, C)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :seq_len, :]

        t_emb = self.time_embed(timesteps)

        # Expand melody from (B, W, d) to (B, H*W, d): each timeframe's melody
        # applies to all H frequency bins. Sequence order is h-major so we
        # repeat the full W melody H times.
        melody_seq = melody_emb.repeat(1, H, 1)

        for i, block in enumerate(self.blocks):
            if i < self.n_control_blocks:
                ctrl_out = self.control_blocks[i](x + melody_seq, t_emb, text_emb)
                ctrl_out = self.zero_linears[i](ctrl_out)
                x = block(x, t_emb, text_emb) + ctrl_out
            else:
                x = block(x, t_emb, text_emb)

        x = self.final_norm(x)
        x = self.output_proj(x)
        return x.permute(0, 2, 1).reshape(B, C, H, W)
