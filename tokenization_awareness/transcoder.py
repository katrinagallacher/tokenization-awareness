"""Load a layer-0 transcoder and encode MLP inputs into sparse features.

A transcoder is a small model trained to reproduce one MLP layer's output
through a wide, sparse, mostly-interpretable bottleneck. Qwen3-4B's layer-0
MLP takes a 2,560-dimensional residual-stream vector; the transcoder
re-expresses it as ~164k sparse features, one of which (118230) is the subject
of this repo.

Only the encoder is needed here: the question is *how strongly a feature fires*,
not how to reconstruct the MLP output, so ``W_dec`` and ``b_dec`` are loaded but
unused.
"""

from __future__ import annotations

from dataclasses import dataclass

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download

from tokenization_awareness.config import LAYER, TRANSCODER_REPO


def default_device() -> str:
    """``"cuda"`` when a GPU is visible, otherwise ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Transcoder:
    """The encoder half of a transcoder: ``relu(x @ W_enc.T + b_enc)``."""

    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor | None = None
    b_dec: torch.Tensor | None = None

    @property
    def dtype(self) -> torch.dtype:
        return self.W_enc.dtype

    @property
    def n_features(self) -> int:
        return self.W_enc.shape[0]

    @property
    def d_model(self) -> int:
        return self.W_enc.shape[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map MLP-input activations to sparse feature activations.

        Args:
            x: ``[batch, seq_len, d_model]`` activations, any float dtype.

        Returns:
            ``[batch, seq_len, n_features]`` non-negative feature activations.
        """
        x = x.to(self.dtype)
        return torch.nn.functional.relu(x @ self.W_enc.T + self.b_enc)


def load_transcoder(
    layer: int = LAYER,
    repo_id: str = TRANSCODER_REPO,
    device: str | None = None,
) -> Transcoder:
    """Download (and cache) one layer's transcoder weights and wrap them.

    The weights keep their published dtype — bfloat16 — which is also the dtype
    the model is run in, so no precision is silently gained or lost in between.
    """
    device = device or default_device()
    path = hf_hub_download(repo_id=repo_id, filename=f"layer_{layer}.safetensors")
    weights = safetensors.torch.load_file(path)

    return Transcoder(
        W_enc=weights["W_enc"].to(device),
        b_enc=weights["b_enc"].to(device),
        W_dec=weights["W_dec"].to(device),
        b_dec=weights["b_dec"].to(device),
    )
