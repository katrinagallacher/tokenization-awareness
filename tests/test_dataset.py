"""Tests for the dataset stage.

The tests marked ``slow`` download the Qwen3-4B tokenizer (a few MB, cached
afterwards) and assert the exact group counts reported in the write-up, so a
refactor that quietly changes the dataset gets caught.

    pytest                    # everything
    pytest -m "not slow"      # skip the ones needing the Hub
"""

from __future__ import annotations

import pytest

from tokenization_awareness.dataset import (
    CompoundGroups,
    build_groups,
    filter_closed_compounds,
    load_compounds,
    load_tokenizer,
    tokenize_both_ways,
)

# Counts from the reference run, reproduced exactly by the refactored pipeline.
EXPECTED = {
    "source": 1720,
    "closed": 1566,
    "true_single": 82,
    "conditional_single": 390,
    "true_multi": 1093,
    "bare_single_only": 1,
}


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def compounds():
    return load_compounds()


def test_filter_drops_open_and_hyphenated():
    words = ["airport", "able-bodied", "night shift", "flowerpot"]
    assert filter_closed_compounds(words) == ["airport", "flowerpot"]


def test_filter_keeps_order_and_duplicates():
    words = ["airport", "airport", "sunflower"]
    assert filter_closed_compounds(words) == words


def test_groups_are_disjoint_and_total():
    groups = CompoundGroups(
        true_single=["a"],
        conditional_single=["b", "c"],
        true_multi=["d"],
        bare_single_only=["e"],
        source_count=5,
        closed_count=5,
    )
    members = [w for words in groups.as_dict().values() for w in words]
    assert len(members) == len(set(members))
    assert len(members) + len(groups.bare_single_only) == groups.closed_count


def test_spaced_prepends_one_space():
    groups = CompoundGroups(true_single=["airport", "football"])
    assert groups.spaced("true_single") == [" airport", " football"]


def test_source_list_is_intact(compounds):
    assert len(compounds) == EXPECTED["source"]
    assert all(word == word.strip() for word in compounds)


@pytest.mark.slow
def test_canonical_examples_tokenize_as_documented(tokenizer):
    """The three worked examples the write-up is built around."""
    bare, spaced = tokenize_both_ways(tokenizer, "newsletter")
    assert len(bare) == 1 and len(spaced) == 1  # true single

    bare, spaced = tokenize_both_ways(tokenizer, "firefighter")
    assert len(bare) == 2 and len(spaced) == 1  # conditional single

    bare, spaced = tokenize_both_ways(tokenizer, "sunflower")
    assert len(bare) == 2 and len(spaced) == 2  # true multi


@pytest.mark.slow
def test_group_counts_match_reference_run(tokenizer, compounds):
    counts = build_groups(tokenizer, compounds).counts()
    assert counts == EXPECTED


@pytest.mark.slow
def test_every_closed_compound_lands_in_exactly_one_group(tokenizer, compounds):
    groups = build_groups(tokenizer, compounds)
    placed = sum(len(words) for words in groups.as_dict().values())
    assert placed + len(groups.bare_single_only) == groups.closed_count


@pytest.mark.slow
def test_capitalization_shrinks_the_conditional_group(tokenizer, compounds):
    """Capitalizing removes most of the whitespace-conditional merges."""
    as_written = build_groups(tokenizer, compounds).counts()
    capitalized = build_groups(tokenizer, compounds, transform=str.capitalize).counts()
    assert capitalized["conditional_single"] < as_written["conditional_single"] / 3
