"""Measure how strongly one transcoder feature fires on a list of words.

Each word is run through the model on its own, with no surrounding context, and
the feature activation is read at the **last token position**. For a word split
into several tokens that last position is the one that has seen all the others,
so it is where a detokenization signal would have to show up.

The hook point is ``blocks.{layer}.ln2.hook_normalized`` — the layer-normalised
residual stream *entering* the MLP, which is exactly what the transcoder was
trained to take as input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformer_lens import HookedTransformer

from tokenization_awareness.config import FEATURE_INDEX, LAYER, MODEL_NAME
from tokenization_awareness.transcoder import Transcoder, default_device


def load_model(
    model_name: str = MODEL_NAME,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> HookedTransformer:
    """Load the model under study with TransformerLens hooks attached.

    This is the step that needs a GPU. Qwen3-4B in bfloat16 wants roughly 9 GB
    of VRAM; it also runs on CPU, slowly.
    """
    return HookedTransformer.from_pretrained(
        model_name,
        dtype=dtype,
        device=device or default_device(),
    )


def hook_name(layer: int = LAYER) -> str:
    """Name of the activation the transcoder consumes."""
    return f"blocks.{layer}.ln2.hook_normalized"


@torch.no_grad()
def measure_activations(
    model: HookedTransformer,
    transcoder: Transcoder,
    words: Sequence[str],
    *,
    layer: int = LAYER,
    feature_index: int = FEATURE_INDEX,
    progress: bool = True,
) -> dict[str, float]:
    """Feature activation at the last token of each word.

    Args:
        model: hooked model.
        transcoder: the layer's transcoder (only its encoder is used).
        words: words to measure, **including any leading space** — spacing is
            the independent variable here, so it is never added silently.
        layer: layer whose MLP input is read.
        feature_index: which transcoder feature to report.
        progress: print a counter every 100 words.

    Returns:
        ``{word: activation}``, preserving input order.
    """
    name = hook_name(layer)
    activations: dict[str, float] = {}

    for index, word in enumerate(words, start=1):
        input_ids = model.tokenizer(word, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(model.cfg.device)

        _, cache = model.run_with_cache(input_ids, names_filter=[name])
        features = transcoder.encode(cache[name])
        activations[word] = float(features[0, -1, feature_index].cpu().item())

        if progress and index % 100 == 0:
            print(f"  {index}/{len(words)} words", flush=True)

    return activations


def measure_groups(
    model: HookedTransformer,
    transcoder: Transcoder,
    groups: dict[str, Iterable[str]],
    *,
    layer: int = LAYER,
    feature_index: int = FEATURE_INDEX,
) -> dict[str, dict[str, float]]:
    """Run :func:`measure_activations` over several named groups."""
    results: dict[str, dict[str, float]] = {}
    for group_name, words in groups.items():
        words = list(words)
        print(f"Measuring {group_name} ({len(words)} words)...", flush=True)
        results[group_name] = measure_activations(
            model,
            transcoder,
            words,
            layer=layer,
            feature_index=feature_index,
        )
    return results


def save_activations(results: dict[str, dict[str, float]], path: str | Path) -> None:
    """Write measured activations to JSON so analysis can run without a GPU."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)


def load_activations(path: str | Path) -> dict[str, dict[str, float]]:
    """Read activations written by :func:`save_activations`."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
