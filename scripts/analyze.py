#!/usr/bin/env python3
"""Stage 3 — statistics and figures from the measured activations.

No GPU needed: this reads ``activations.json`` and produces the two figures
plus the numbers quoted in the write-up.

    python scripts/analyze.py
    python scripts/analyze.py --equal-var   # reproduce the original notebook's test
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running straight from a clone, without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402

from tokenization_awareness.activations import load_activations  # noqa: E402
from tokenization_awareness.analysis import (  # noqa: E402
    compare_groups,
    describe_all,
    format_nonzero,
    frequency_correlations,
    frequency_scores,
)
from tokenization_awareness.config import (  # noqa: E402
    DEFAULT_FIGURES_DIR,
    DEFAULT_RESULTS_DIR,
)
from tokenization_awareness.plots import (  # noqa: E402
    plot_activation_by_group,
    plot_frequency_vs_activation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-a",
        "--activations",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "activations.json",
        help="output of measure_activations.py",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="directory to write figures into",
    )
    parser.add_argument(
        "--equal-var",
        action="store_true",
        help="use Student's t-test instead of Welch's, as the original notebook did",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="print statistics only",
    )
    return parser.parse_args()


def rule(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def main() -> None:
    args = parse_args()

    if not args.activations.exists():
        raise SystemExit(
            f"{args.activations} not found — run scripts/measure_activations.py first."
        )

    activations = load_activations(args.activations)

    rule("Descriptive statistics")
    for group_stats in describe_all(activations).values():
        print(group_stats)
        print()

    rule("Group comparisons")
    test_name = "Student" if args.equal_var else "Welch"
    print(f"({test_name}'s t-test)\n")
    for comparison in compare_groups(activations, equal_var=args.equal_var):
        print(f"  {comparison}")

    rule("Words that fired")
    for name, values in activations.items():
        print(format_nonzero(values, name, limit=20))
        print()

    frequencies = {
        name: frequency_scores(list(values)) for name, values in activations.items()
    }

    rule("Frequency vs activation, within groups")
    for correlation in frequency_correlations(activations, frequencies):
        print(f"  {correlation}")

    if not args.no_figures:
        rule("Figures")
        plot_activation_by_group(
            activations, path=args.figures / "activation_by_token_type.png"
        )
        plot_frequency_vs_activation(
            activations,
            frequencies,
            path=args.figures / "frequency_vs_activation.png",
        )


if __name__ == "__main__":
    main()
