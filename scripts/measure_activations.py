#!/usr/bin/env python3
"""Stage 2 — measure feature 118230's activation on every compound.

This is the step that needs the model. Qwen3-4B in bfloat16 fits in ~9 GB of
VRAM; a free Colab T4 runs the full 1,565-word sweep in around 15 minutes.

    python scripts/measure_activations.py
    python scripts/measure_activations.py --feature 118230 --layer 0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running straight from a clone, without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402

from tokenization_awareness.activations import (  # noqa: E402
    load_model,
    measure_groups,
    save_activations,
)
from tokenization_awareness.config import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    FEATURE_INDEX,
    LAYER,
)
from tokenization_awareness.transcoder import load_transcoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-g",
        "--groups",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "groups.json",
        help="output of build_dataset.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "activations.json",
        help="where to write the measured activations",
    )
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--feature", type=int, default=FEATURE_INDEX)
    parser.add_argument(
        "--device",
        default=None,
        help="force a device; defaults to cuda when available",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="measure only the first N words per group (for a smoke test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.groups.exists():
        raise SystemExit(
            f"{args.groups} not found — run scripts/build_dataset.py first."
        )

    with open(args.groups, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    # Activations are always measured on the spaced form: that is how a word
    # appears in running text, and it is the form that makes the conditional
    # group a single token.
    groups = {
        name: [" " + word for word in (words[: args.limit] if args.limit else words)]
        for name, words in payload["groups"].items()
    }

    print(f"Loading model (layer {args.layer}, feature {args.feature})...")
    model = load_model(device=args.device)
    transcoder = load_transcoder(layer=args.layer, device=args.device)
    print(
        f"Transcoder: {transcoder.n_features} features x "
        f"{transcoder.d_model} dims, dtype {transcoder.dtype}"
    )

    results = measure_groups(
        model,
        transcoder,
        groups,
        layer=args.layer,
        feature_index=args.feature,
    )

    save_activations(results, args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
