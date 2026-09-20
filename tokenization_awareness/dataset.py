"""Build the compound-word dataset and split it by tokenization stability.

The whole experiment rests on one observation: for a lot of English compound
words, whether the word is a single token depends on whether it is preceded by
a space. ``firefighter`` is two tokens; ``" firefighter"`` is one.

This module turns a flat word list into three groups:

``true_single``
    One token with *and* without the leading space (``newsletter``).
``conditional_single``
    One token *only* with the leading space (``firefighter``).
``true_multi``
    Several tokens either way (``sunflower``).

A fourth, near-empty group (``bare_single_only``) catches the mirror case —
one token without the space, several with it. It exists so nothing is silently
dropped; in Qwen3-4B it holds a single word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tokenization_awareness.config import DEFAULT_COMPOUNDS_PATH, MODEL_NAME


def load_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizerBase:
    """Load the tokenizer of the model under study.

    Only the tokenizer is needed to build the dataset, so this step runs
    on a laptop in seconds — no GPU, no model weights.
    """
    return AutoTokenizer.from_pretrained(model_name)


def load_compounds(path: str | Path = DEFAULT_COMPOUNDS_PATH) -> list[str]:
    """Read the raw compound list, one word per line, blank lines skipped.

    The bundled list holds 1,720 English compounds from
    https://www.proofreadingservices.com/pages/compound-words-list
    """
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def filter_closed_compounds(compounds: Iterable[str]) -> list[str]:
    """Keep only *closed* compounds — written as one solid word.

    Hyphenated (``able-bodied``) and open (``night shift``) compounds are out of
    scope: they contain a character the pre-tokenizer splits on anyway, so the
    single-vs-multi-token question does not arise for them.
    """
    return [word for word in compounds if " " not in word and "-" not in word]


def token_count(
    tokenizer: PreTrainedTokenizerBase,
    word: str,
    *,
    leading_space: bool = False,
) -> int:
    """Number of tokens ``word`` occupies, optionally with a leading space."""
    return len(tokenizer.tokenize(" " + word if leading_space else word))


def tokenize_both_ways(
    tokenizer: PreTrainedTokenizerBase, word: str
) -> tuple[list[str], list[str]]:
    """Return ``(tokens_without_space, tokens_with_space)`` for one word.

    Byte-level markers are decoded back to readable text, so a leading space
    shows up as an actual space rather than ``Ġ``.
    """
    bare = tokenizer.tokenize(word)
    spaced = tokenizer.tokenize(" " + word)
    return (
        [tokenizer.convert_tokens_to_string([t]) for t in bare],
        [tokenizer.convert_tokens_to_string([t]) for t in spaced],
    )


def split_by_token_count(
    tokenizer: PreTrainedTokenizerBase,
    compounds: Iterable[str],
    *,
    leading_space: bool = False,
) -> tuple[list[str], list[str]]:
    """Split words into ``(single_token, multi_token)`` under one spacing regime."""
    single: list[str] = []
    multi: list[str] = []
    for word in compounds:
        if token_count(tokenizer, word, leading_space=leading_space) == 1:
            single.append(word)
        else:
            multi.append(word)
    return single, multi


@dataclass(frozen=True)
class CompoundGroups:
    """The three tokenization-stability groups, plus bookkeeping.

    Words are stored *without* the leading space. Add it back with
    :meth:`spaced` when feeding the model — activations are always measured on
    the spaced form, since that is how a word appears in running text.
    """

    true_single: list[str] = field(default_factory=list)
    conditional_single: list[str] = field(default_factory=list)
    true_multi: list[str] = field(default_factory=list)
    bare_single_only: list[str] = field(default_factory=list)
    source_count: int = 0
    closed_count: int = 0

    def spaced(self, group: str) -> list[str]:
        """The named group with a leading space prepended to every word."""
        return [" " + word for word in getattr(self, group)]

    def as_dict(self) -> dict[str, list[str]]:
        """The three analysis groups, keyed by name."""
        return {
            "true_single": self.true_single,
            "conditional_single": self.conditional_single,
            "true_multi": self.true_multi,
        }

    def counts(self) -> dict[str, int]:
        """Group sizes, including the leftover group and the input totals."""
        return {
            "source": self.source_count,
            "closed": self.closed_count,
            "true_single": len(self.true_single),
            "conditional_single": len(self.conditional_single),
            "true_multi": len(self.true_multi),
            "bare_single_only": len(self.bare_single_only),
        }

    def summary(self) -> str:
        """A short human-readable report of how the dataset came out."""
        counts = self.counts()
        closed = counts["closed"] or 1
        lines = [
            f"{counts['source']:>6} compounds in the source list",
            f"{counts['closed']:>6} closed compounds kept",
            "",
        ]
        for name in ("true_single", "conditional_single", "true_multi"):
            share = 100 * counts[name] / closed
            lines.append(f"{counts[name]:>6} {name:<20} ({share:4.1f}%)")
        if counts["bare_single_only"]:
            lines.append(
                f"{counts['bare_single_only']:>6} bare_single_only     "
                f"(single without space, split with it): "
                f"{', '.join(self.bare_single_only)}"
            )
        return "\n".join(lines)


def build_groups(
    tokenizer: PreTrainedTokenizerBase,
    compounds: Sequence[str],
    *,
    transform: Callable[[str], str] | None = None,
    already_filtered: bool = False,
) -> CompoundGroups:
    """Sort compounds into tokenization-stability groups.

    Args:
        tokenizer: tokenizer of the model under study.
        compounds: raw word list.
        transform: optional per-word transform applied before tokenizing —
            pass ``str.capitalize`` to reproduce the capitalization experiment.
        already_filtered: skip the closed-compound filter if the caller has
            done it.

    Group membership is decided per word from its two token counts, so the
    groups are disjoint by construction and input order is preserved. (The
    original notebook used set differences, which made ordering
    non-deterministic between runs.)
    """
    source_count = len(compounds)
    closed = list(compounds) if already_filtered else filter_closed_compounds(compounds)
    words = [transform(word) for word in closed] if transform else closed

    groups: dict[str, list[str]] = {
        "true_single": [],
        "conditional_single": [],
        "true_multi": [],
        "bare_single_only": [],
    }

    for word in words:
        bare_is_single = token_count(tokenizer, word) == 1
        spaced_is_single = token_count(tokenizer, word, leading_space=True) == 1

        if bare_is_single and spaced_is_single:
            groups["true_single"].append(word)
        elif spaced_is_single:
            groups["conditional_single"].append(word)
        elif bare_is_single:
            groups["bare_single_only"].append(word)
        else:
            groups["true_multi"].append(word)

    return CompoundGroups(
        true_single=groups["true_single"],
        conditional_single=groups["conditional_single"],
        true_multi=groups["true_multi"],
        bare_single_only=groups["bare_single_only"],
        source_count=source_count,
        closed_count=len(closed),
    )
