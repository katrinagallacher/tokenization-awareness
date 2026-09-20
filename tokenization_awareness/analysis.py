"""Statistics over the measured activations.

Three things get asked of the numbers:

1. Do the groups differ? (means, standard deviations, Welch t-tests)
2. Which individual words fire? (the feature is sparse — most words give
   exactly zero, so the non-zero words *are* the signal, not noise to trim)
3. Does frequency explain it? (Zipf score vs activation, per group)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats
from wordfreq import zipf_frequency

from tokenization_awareness.config import FREQUENCY_LANGUAGE, GROUP_LABELS


@dataclass(frozen=True)
class GroupStats:
    """Descriptive statistics for one group."""

    name: str
    n: int
    mean: float
    std: float
    median: float
    n_nonzero: int
    max: float

    @property
    def pct_nonzero(self) -> float:
        return 100 * self.n_nonzero / self.n if self.n else 0.0

    def __str__(self) -> str:
        label = GROUP_LABELS.get(self.name, self.name)
        return (
            f"{label} (n={self.n})\n"
            f"  mean {self.mean:.4f} | std {self.std:.4f} | max {self.max:.4f}\n"
            f"  non-zero: {self.n_nonzero}/{self.n} ({self.pct_nonzero:.1f}%)"
        )


@dataclass(frozen=True)
class Comparison:
    """A two-group Welch t-test."""

    group_a: str
    group_b: str
    t: float
    p: float

    @property
    def direction(self) -> str:
        return ">" if self.t > 0 else "<"

    def __str__(self) -> str:
        a = GROUP_LABELS.get(self.group_a, self.group_a)
        b = GROUP_LABELS.get(self.group_b, self.group_b)
        return f"{a} {self.direction} {b}: t={self.t:.4f}, p={self.p:.4e}"


def describe(name: str, values: list[float]) -> GroupStats:
    """Descriptive statistics for one group's activations."""
    array = np.asarray(values, dtype=float)
    return GroupStats(
        name=name,
        n=len(array),
        mean=float(array.mean()) if array.size else 0.0,
        std=float(array.std()) if array.size else 0.0,
        median=float(np.median(array)) if array.size else 0.0,
        n_nonzero=int((array > 0).sum()),
        max=float(array.max()) if array.size else 0.0,
    )


def describe_all(activations: dict[str, dict[str, float]]) -> dict[str, GroupStats]:
    """Descriptive statistics for every group."""
    return {
        name: describe(name, list(values.values()))
        for name, values in activations.items()
    }


def compare_groups(
    activations: dict[str, dict[str, float]],
    *,
    equal_var: bool = False,
) -> list[Comparison]:
    """Welch t-tests between every pair of groups.

    Welch rather than Student: the groups differ in size by more than an order
    of magnitude and their variances are visibly unequal, which is exactly the
    case Student's t-test handles badly. (The original notebook used the
    equal-variance default; the conclusions do not change, but Welch is the
    honest test here. Pass ``equal_var=True`` to reproduce the original run.)
    """
    results: list[Comparison] = []
    for name_a, name_b in combinations(activations, 2):
        values_a = list(activations[name_a].values())
        values_b = list(activations[name_b].values())
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=equal_var)
        results.append(Comparison(name_a, name_b, float(t_stat), float(p_value)))
    return results


def nonzero_words(
    activations: dict[str, float], *, threshold: float = 0.0
) -> list[tuple[str, float]]:
    """Words that fire at all, strongest first.

    The feature is sparse: in most groups the large majority of words sit at
    exactly zero, so "outlier" here means "the feature actually fired", not
    "suspicious data point".
    """
    firing = {word: value for word, value in activations.items() if value > threshold}
    return sorted(firing.items(), key=lambda pair: pair[1], reverse=True)


def format_nonzero(
    activations: dict[str, float], name: str, *, limit: int | None = None
) -> str:
    """A printable table of the words that fired in one group."""
    firing = nonzero_words(activations)
    total = len(activations)
    label = GROUP_LABELS.get(name, name)

    if not firing:
        return f"{label}: no non-zero activations out of {total}"

    share = 100 * len(firing) / total
    lines = [
        f"{label}: {len(firing)} non-zero out of {total} ({share:.1f}%)",
        f"  range {firing[-1][1]:.4f} to {firing[0][1]:.4f}",
        "",
        f"  {'word':<24}{'activation':>12}",
        f"  {'-' * 36}",
    ]
    shown = firing[:limit] if limit else firing
    lines += [f"  {word.strip():<24}{value:>12.4f}" for word, value in shown]
    if limit and len(firing) > limit:
        lines.append(f"  ... and {len(firing) - limit} more")
    return "\n".join(lines)


def frequency_scores(
    words: list[str], language: str = FREQUENCY_LANGUAGE
) -> dict[str, float]:
    """Zipf frequency for each word, on a 0-8 scale.

    Roughly: below 3 is rare, 3-4 common, 5+ very common. Leading spaces are
    stripped before lookup, since ``wordfreq`` knows nothing about tokenizers.
    """
    return {word: zipf_frequency(word.strip(), language) for word in words}


@dataclass(frozen=True)
class Correlation:
    """Pearson correlation between Zipf frequency and activation, per group."""

    name: str
    n: int
    r: float
    p: float

    def __str__(self) -> str:
        label = GROUP_LABELS.get(self.name, self.name)
        return f"{label} (n={self.n}): r={self.r:.4f}, p={self.p:.4e}"


def frequency_correlations(
    activations: dict[str, dict[str, float]],
    frequencies: dict[str, dict[str, float]],
) -> list[Correlation]:
    """Correlate frequency against activation within each group.

    A flat correlation inside a group, combined with a strong difference
    *between* groups, is the result that rules out "the feature just tracks how
    common the word is".
    """
    results: list[Correlation] = []
    for name, values in activations.items():
        freqs = frequencies.get(name, {})
        paired = [(freqs[word], values[word]) for word in values if word in freqs]
        if len(paired) < 2:
            continue
        x = np.array([f for f, _ in paired])
        y = np.array([a for _, a in paired])
        if x.std() == 0 or y.std() == 0:
            results.append(Correlation(name, len(paired), 0.0, 1.0))
            continue
        r, p = stats.pearsonr(x, y)
        results.append(Correlation(name, len(paired), float(r), float(p)))
    return results
