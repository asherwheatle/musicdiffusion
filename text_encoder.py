"""Character-level text encoder for mood descriptions."""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Learned character-level text encoder for mood descriptions.

    Produces a sequence of text embeddings for cross-attention conditioning
    in the DiT. Self-contained — no pretrained model downloads needed.
    For production quality, swap in T5-base (paper's choice).
    """

    def __init__(self, d_model=256, max_len=128, n_layers=2, n_heads=4):
        super().__init__()
        self.max_len = max_len
        self.char_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def tokenize(text: str, max_len: int = 128) -> torch.LongTensor:
        tokens = [min(ord(c), 255) for c in text[:max_len]]
        tokens += [0] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, seq_len) character indices 0-255
        Returns:
            (B, seq_len, d_model) text embeddings for cross-attention
        """
        B, S = tokens.shape
        pos = torch.arange(S, device=tokens.device)
        x = self.char_embed(tokens) + self.pos_embed(pos)
        x = self.encoder(x)
        return self.norm(x)
