"""The two figures from the write-up.

Both take the same ``{group: {word: activation}}`` structure the rest of the
pipeline passes around, and both return the Matplotlib figure so a caller can
tweak it before saving.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display

import matplotlib.pyplot as plt  # noqa: E402

from tokenization_awareness.config import (  # noqa: E402
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_NAMES,
)


def _ordered(activations: dict[str, dict[str, float]]) -> list[str]:
    """Group names in the canonical reporting order, ignoring absent ones."""
    return [name for name in GROUP_NAMES if name in activations]


def plot_activation_by_group(
    activations: dict[str, dict[str, float]],
    *,
    path: str | Path | None = None,
    title: str = "Feature activation by tokenization group",
) -> plt.Figure:
    """Box plot of activation per group, with means marked.

    Medians are hidden and means shown instead: the feature is sparse enough
    that two of the three medians sit at zero, which tells you the distribution
    is zero-inflated but nothing about the difference between groups.
    """
    names = _ordered(activations)
    data = [list(activations[name].values()) for name in names]
    labels = [f"{GROUP_LABELS[name]}\n(n={len(activations[name])})" for name in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanline=True,
        meanprops=dict(color="black", linewidth=2, linestyle="-"),
        medianprops=dict(visible=False),
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        flierprops=dict(
            marker="o",
            markerfacecolor="gray",
            markersize=4,
            linestyle="none",
            markeredgecolor="black",
            alpha=0.5,
        ),
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Activation value", fontsize=12)
    ax.set_xlabel("Tokenization group", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.set_axisbelow(True)
    ax.legend(
        handles=[plt.Line2D([0], [0], color="black", linewidth=2, label="Mean")],
        loc="upper right",
    )

    fig.tight_layout()
    if path:
        _save(fig, path)
    return fig


def plot_frequency_vs_activation(
    activations: dict[str, dict[str, float]],
    frequencies: dict[str, dict[str, float]],
    *,
    path: str | Path | None = None,
    label_top_n: int = 10,
    title: str = "Word frequency vs feature activation",
) -> plt.Figure:
    """Scatter of Zipf frequency against activation, coloured by group.

    The strongest ``label_top_n`` words per group are annotated — those are the
    words worth arguing about, and naming them lets a reader check the claim
    against their own intuitions.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    for name in _ordered(activations):
        values = activations[name]
        freqs = frequencies.get(name, {})
        paired = [(freqs[word], values[word], word) for word in values if word in freqs]
        if not paired:
            continue

        x = [f for f, _, _ in paired]
        y = [a for _, a, _ in paired]
        ax.scatter(
            x,
            y,
            alpha=0.7,
            color=GROUP_COLORS[name],
            s=60,
            edgecolors="white",
            linewidth=0.8,
            label=f"{GROUP_LABELS[name]} (n={len(paired)})",
        )

        for freq, act, word in sorted(paired, key=lambda p: p[1], reverse=True)[
            :label_top_n
        ]:
            ax.annotate(
                word.strip(),
                xy=(freq, act),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=13,
                color=GROUP_COLORS[name],
                alpha=0.85,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=GROUP_COLORS[name],
                    alpha=0.7,
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=0",
                    color=GROUP_COLORS[name],
                    alpha=0.5,
                    lw=0.8,
                ),
            )

    ax.set_xlabel("Zipf frequency score", fontsize=14)
    ax.set_ylabel("Feature activation", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="black", fontsize=13)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    if path:
        _save(fig, path)
    return fig


def _save(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Wrote {path}")
