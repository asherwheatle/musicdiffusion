"""Text encoders for mood descriptions.

Two encoders live here:

  * ClapTextEncoder (Lever A, default): a *frozen* CLAP text tower followed by
    a small trainable projection. CLAP was contrastively trained on text<->audio
    pairs, so a prompt like "dark and mysterious" already lands near
    dark/mysterious-sounding audio — and unseen phrasings ("eerie", "ominous
    film score") generalize because CLAP knows they are semantically close.
    This is the encoder the pipeline now trains and infers with, and it uses the
    *same* CLAP representation the evaluation judges against.

  * TextEncoder (legacy): a character-level transformer trained from scratch.
    It only ever sees the five fixed mood strings during training, so it learns
    a lookup from those exact character sequences to audio — no semantics, no
    generalization. Kept for reference / ablation only.
"""

import os

import torch
import torch.nn as nn


def load_clap_text_model(clap_ckpt: str = None, device: str = "cpu"):
    """Load a LAION-CLAP module (music checkpoint) for text embedding.

    Mirrors evaluate.Clap's loading so training and evaluation share one CLAP
    representation. Returns the frozen CLAP_Module; the caller wraps it.
    """
    try:
        import laion_clap
    except ImportError as e:
        raise ImportError(
            "laion_clap is not installed. Run:  uv pip install laion-clap\n"
            "and download the music checkpoint:\n"
            "  wget https://huggingface.co/lukewys/laion_clap/resolve/main/"
            "music_audioset_epoch_15_esc_90.14.pt"
        ) from e
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base",
                                   device=device)
    if clap_ckpt and os.path.exists(clap_ckpt):
        print(f"[CLAP-TXT] Loading music checkpoint: {clap_ckpt}")
        model.load_ckpt(clap_ckpt)
    else:
        print("[CLAP-TXT] WARNING: no music checkpoint given, loading the "
              "default general-audio checkpoint (weaker at musical mood).")
        model.load_ckpt()
    model.eval()
    return model


class ClapTextEncoder(nn.Module):
    """Frozen CLAP text tower + a trainable projection into a DiT
    cross-attention sequence.

        prompt --(frozen CLAP text tower)--> (clap_dim,) L2-normed
               --(trainable Linear)--------> (n_tokens, d_model) sequence

    Only the projection (`proj` + `norm`) is trainable and saved in the state
    dict; CLAP is frozen and deliberately held OUTSIDE nn.Module registration
    (in a plain list) so it does not appear in `.parameters()` or
    `.state_dict()` and is not toggled by `.train()/.eval()`.

    Usage matches the old TextEncoder except CLAP does its own tokenization, so
    callers pass raw strings through `.encode(...)`:

        clap_emb = text_enc.encode(["dark and mysterious"])   # (B, clap_dim)
        text_emb = text_enc(clap_emb)                         # (B, n_tokens, d)

    `encode` (frozen, no grad) is cheap to precompute once per unique prompt.
    """

    def __init__(self, d_model: int, clap_model=None, clap_ckpt: str = None,
                 n_tokens: int = 4, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens

        clap = (clap_model if clap_model is not None
                else load_clap_text_model(clap_ckpt, device))
        clap.eval()
        for p in clap.parameters():
            p.requires_grad_(False)
        self._clap = [clap]            # hidden from parameters()/state_dict()
        self.device = device

        # Infer CLAP's text-embedding dim from a probe encode so the projection
        # shape is always correct (HTSAT-base music checkpoint -> 512).
        with torch.no_grad():
            self.clap_dim = int(self._encode_raw([""]).shape[-1])

        self.proj = nn.Linear(self.clap_dim, d_model * n_tokens)
        self.norm = nn.LayerNorm(d_model)

    @property
    def clap(self):
        return self._clap[0]

    @torch.no_grad()
    def _encode_raw(self, texts) -> torch.Tensor:
        emb = self.clap.get_text_embedding(list(texts), use_tensor=True)
        emb = emb.detach().float()
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)   # L2-norm
        return emb.to(self.device)

    @torch.no_grad()
    def encode(self, texts, chunk: int = 256) -> torch.Tensor:
        """Frozen CLAP text embeddings for a list of strings -> (B, clap_dim).

        Chunked so a large corpus (one string per training song) doesn't push
        the whole batch through CLAP's text tower at once.
        """
        texts = list(texts)
        if len(texts) <= chunk:
            return self._encode_raw(texts)
        return torch.cat([self._encode_raw(texts[i:i + chunk])
                          for i in range(0, len(texts), chunk)], dim=0)

    def forward(self, clap_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clap_emb: (B, clap_dim) frozen, L2-normed CLAP text embeddings
                      (from `.encode`).
        Returns:
            (B, n_tokens, d_model) conditioning sequence for cross-attention.
        """
        B = clap_emb.shape[0]
        h = self.proj(clap_emb).view(B, self.n_tokens, self.d_model)
        return self.norm(h)


class TextEncoder(nn.Module):
    """
    Legacy learned character-level text encoder for mood descriptions.

    Produces a sequence of text embeddings for cross-attention conditioning
    in the DiT. Self-contained — no pretrained model downloads needed, but it
    learns no word semantics (see module docstring). Superseded by
    ClapTextEncoder; kept for reference / ablation.
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
