"""Tests for the statistics layer, on synthetic activations.

These never touch the model: the point is that the summaries and tests behave
sensibly on a sparse, zero-inflated distribution, which is what the real
measurements look like.
"""

from __future__ import annotations

from tokenization_awareness.analysis import (
    compare_groups,
    describe,
    describe_all,
    format_nonzero,
    frequency_correlations,
    frequency_scores,
    nonzero_words,
)

SPARSE = {
    "true_single": {" heartbeat": 20.25, " airport": 0.0, " football": 0.0},
    "conditional_single": {" lawsuit": 6.44, " waterfall": 14.75, " roadway": 0.0},
    "true_multi": {" flowerpot": 0.0, " snowfall": 0.0, " lighthouse": 9.125},
}


def test_describe_counts_nonzero_not_just_mean():
    stats = describe("true_single", [0.0, 0.0, 0.0, 4.0])
    assert stats.n == 4
    assert stats.n_nonzero == 1
    assert stats.pct_nonzero == 25.0
    assert stats.median == 0.0
    assert stats.max == 4.0


def test_describe_handles_an_all_zero_group():
    stats = describe("true_multi", [0.0, 0.0])
    assert stats.mean == 0.0
    assert stats.n_nonzero == 0
    assert stats.pct_nonzero == 0.0


def test_describe_all_covers_every_group():
    assert set(describe_all(SPARSE)) == set(SPARSE)


def test_nonzero_words_sorted_strongest_first():
    firing = nonzero_words(SPARSE["conditional_single"])
    assert [word for word, _ in firing] == [" waterfall", " lawsuit"]


def test_nonzero_words_excludes_exact_zeros():
    assert all(value > 0 for _, value in nonzero_words(SPARSE["true_multi"]))


def test_format_nonzero_reports_a_share():
    text = format_nonzero(SPARSE["true_multi"], "true_multi")
    assert "1 non-zero out of 3" in text
    assert "lighthouse" in text


def test_format_nonzero_handles_empty_group():
    text = format_nonzero({" a": 0.0, " b": 0.0}, "true_multi")
    assert "no non-zero activations" in text


def test_compare_groups_covers_every_pair():
    comparisons = compare_groups(SPARSE)
    assert len(comparisons) == 3
    pairs = {(c.group_a, c.group_b) for c in comparisons}
    assert ("true_single", "conditional_single") in pairs


def test_frequency_scores_strip_leading_space():
    scores = frequency_scores([" heartbeat", "heartbeat"])
    assert scores[" heartbeat"] == scores["heartbeat"] > 0


def test_frequency_correlations_skip_degenerate_groups():
    activations = {"true_multi": {" a": 0.0}}
    frequencies = {"true_multi": {" a": 3.0}}
    assert frequency_correlations(activations, frequencies) == []


def test_frequency_correlations_run_per_group():
    frequencies = {
        name: frequency_scores(list(values)) for name, values in SPARSE.items()
    }
    results = frequency_correlations(SPARSE, frequencies)
    assert {c.name for c in results} == set(SPARSE)
