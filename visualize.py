"""Spectrogram visualization utilities."""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_mood_edit(original_mel: torch.Tensor, edited_mel: torch.Tensor,
                   mood_text: str, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im0 = axes[0].imshow(original_mel.squeeze().cpu().numpy(),
                         aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Original", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Time Frames")
    axes[0].set_ylabel("Mel Bins")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(edited_mel.squeeze().cpu().numpy(),
                         aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title(f'Edited: "{mood_text}"', fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Mel Bins")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] Saved: {output_path}")
