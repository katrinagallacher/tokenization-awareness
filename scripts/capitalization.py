#!/usr/bin/env python3
"""Does capitalizing a compound change which group it lands in?

Capitalization is a second lever on tokenization, independent of the leading
space: ``Firefighter`` need not be tokenized like ``firefighter``. If the
conditional group shrinks sharply under capitalization, that is more evidence
the grouping is about merge history rather than about anything semantic.

Tokenizer only — no GPU, a few seconds.

    python scripts/capitalization.py
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
    filter_closed_compounds,
    load_compounds,
    load_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compounds", type=Path, default=DEFAULT_COMPOUNDS_PATH)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "capitalization.json",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=12,
        help="how many words that changed group to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = load_tokenizer()
    closed = filter_closed_compounds(load_compounds(args.compounds))

    lower = build_groups(tokenizer, closed, already_filtered=True)
    upper = build_groups(
        tokenizer, closed, transform=str.capitalize, already_filtered=True
    )

    print("As written\n----------")
    print(lower.summary())
    print("\nCapitalized\n-----------")
    print(upper.summary())

    group_of_lower = {
        word: name for name, words in lower.as_dict().items() for word in words
    }
    group_of_upper = {
        word.lower(): name for name, words in upper.as_dict().items() for word in words
    }

    moved: dict[str, list[str]] = {}
    for word, was in group_of_lower.items():
        now = group_of_upper.get(word)
        if now and now != was:
            moved.setdefault(f"{was} -> {now}", []).append(word)

    print("\nGroup changes under capitalization\n" + "-" * 34)
    for transition, words in sorted(moved.items(), key=lambda kv: -len(kv[1])):
        sample = ", ".join(words[: args.examples])
        extra = len(words) - args.examples
        more = f" (+{extra} more)" if extra > 0 else ""
        print(f"  {len(words):>5}  {transition}")
        print(f"         {sample}{more}")

    payload = {
        "counts": {"as_written": lower.counts(), "capitalized": upper.counts()},
        "moved": moved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
