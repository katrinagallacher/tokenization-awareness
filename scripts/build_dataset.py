#!/usr/bin/env python3
"""Stage 1 — sort compounds into tokenization-stability groups.

Needs only the tokenizer, so it runs anywhere in a few seconds.

    python scripts/build_dataset.py
    python scripts/build_dataset.py --capitalize -o results/groups_capitalized.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running straight from a clone, without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402

from tokenization_awareness.config import (  # noqa: E402
    DEFAULT_COMPOUNDS_PATH,
    DEFAULT_RESULTS_DIR,
)
from tokenization_awareness.dataset import (  # noqa: E402
    build_groups,
    load_compounds,
    load_tokenizer,
    tokenize_both_ways,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compounds",
        type=Path,
        default=DEFAULT_COMPOUNDS_PATH,
        help="word list, one compound per line",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "groups.json",
        help="where to write the grouped words",
    )
    parser.add_argument(
        "--capitalize",
        action="store_true",
        help="capitalize every word first (the capitalization experiment)",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="how many worked tokenization examples to print per group",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = load_tokenizer()
    compounds = load_compounds(args.compounds)
    groups = build_groups(
        tokenizer,
        compounds,
        transform=str.capitalize if args.capitalize else None,
    )

    print(groups.summary())

    if args.examples:
        print("\nWorked examples (· marks a leading space):")
        for name, words in groups.as_dict().items():
            print(f"\n  {name}")
            for word in words[: args.examples]:
                bare, spaced = tokenize_both_ways(tokenizer, word)
                bare_str = " | ".join(t.replace(" ", "·") for t in bare)
                spaced_str = " | ".join(t.replace(" ", "·") for t in spaced)
                print(f"    {word:<18} {bare_str:<32} -> {spaced_str}")

    payload = {
        "compounds_path": str(args.compounds),
        "capitalized": args.capitalize,
        "counts": groups.counts(),
        "groups": groups.as_dict(),
        "bare_single_only": groups.bare_single_only,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
