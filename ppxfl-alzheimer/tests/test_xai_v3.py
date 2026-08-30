"""Tests for region attribution and its comparison across models."""

import numpy as np
import pytest

from src.xai_v3 import (
    agreement,
    attribution_map,
    compare,
    medial_temporal_share,
    normalise,
    region_importance,
    top_k_overlap,
)

FEATURE_NAMES = [
    "gm::Left Hippocampus", "csf::Left Hippocampus",
    "gm::Right Hippocampus", "csf::Right Hippocampus",
    "gm::Left Amygdala", "csf::Left Amygdala",
    "gm::Right Amygdala", "csf::Right Amygdala",
    "gm::Brain-Stem", "csf::Brain-Stem",
    "gm::Left Thalamus", "csf::Left Thalamus",
]


def test_region_importance_sums_features_of_the_same_region():
    weights = np.zeros((len(FEATURE_NAMES), 2))
    weights[0] = [1.0, -2.0]   # gm::Left Hippocampus
    weights[1] = [0.5, 0.5]    # csf::Left Hippocampus
    importance = region_importance(weights, FEATURE_NAMES)
    assert importance["Left Hippocampus"] == pytest.approx(4.0)
    assert importance["Brain-Stem"] == pytest.approx(0.0)


def test_region_importance_uses_magnitudes_not_signs():
    positive = np.zeros((len(FEATURE_NAMES), 1))
    positive[0] = 3.0
    negative = -positive
    assert (region_importance(positive, FEATURE_NAMES)
            == region_importance(negative, FEATURE_NAMES))


def test_normalise_makes_shares_sum_to_one():
    shares = normalise({"a": 2.0, "b": 6.0})
    assert shares == {"a": 0.25, "b": 0.75}


def test_normalise_of_an_all_zero_map_stays_zero():
    assert normalise({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}


def test_agreement_is_one_for_identical_rankings():
    mapping = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    assert agreement(mapping, mapping) == pytest.approx(1.0)


def test_agreement_is_minus_one_for_reversed_rankings():
    reference = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    reversed_map = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    assert agreement(reference, reversed_map) == pytest.approx(-1.0)


def test_agreement_needs_enough_shared_regions():
    assert np.isnan(agreement({"a": 1.0}, {"a": 1.0}))


def test_agreement_is_undefined_for_a_flat_map():
    reference = {"a": 1.0, "b": 2.0, "c": 3.0}
    flat = {"a": 1.0, "b": 1.0, "c": 1.0}
    assert np.isnan(agreement(reference, flat))


def test_medial_temporal_share_counts_only_those_regions():
    weights = np.zeros((len(FEATURE_NAMES), 1))
    weights[0] = 1.0   # Left Hippocampus
    weights[8] = 3.0   # Brain-Stem
    share = medial_temporal_share(region_importance(weights, FEATURE_NAMES))
    assert share == pytest.approx(0.25)


def test_medial_temporal_share_is_one_when_all_mass_is_there():
    weights = np.zeros((len(FEATURE_NAMES), 1))
    weights[0] = 2.0
    weights[4] = 2.0
    share = medial_temporal_share(region_importance(weights, FEATURE_NAMES))
    assert share == pytest.approx(1.0)


def test_top_k_overlap_is_one_for_the_same_ranking():
    mapping = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert top_k_overlap(mapping, mapping, k=2) == pytest.approx(1.0)


def test_top_k_overlap_is_zero_for_disjoint_tops():
    reference = {"a": 3.0, "b": 2.0, "c": 0.0, "d": 0.0}
    candidate = {"a": 0.0, "b": 0.0, "c": 3.0, "d": 2.0}
    assert top_k_overlap(reference, candidate, k=2) == pytest.approx(0.0)


def test_top_k_overlap_of_empty_maps_is_undefined():
    assert np.isnan(top_k_overlap({}, {}, k=3))


def test_compare_reports_every_field():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(len(FEATURE_NAMES), 3))
    candidate = reference + rng.normal(scale=0.05, size=reference.shape)
    result = compare(reference, candidate, FEATURE_NAMES, k=3)
    assert set(result) == {
        "agreement_spearman", "top_k_overlap", "medial_temporal_share",
        "reference_medial_temporal_share", "top_regions",
    }
    assert result["agreement_spearman"] > 0.5
    assert len(result["top_regions"]) == 3


def test_compare_of_a_model_with_itself_agrees_perfectly():
    weights = np.arange(len(FEATURE_NAMES) * 2, dtype=float).reshape(-1, 2)
    result = compare(weights, weights, FEATURE_NAMES, k=4)
    assert result["agreement_spearman"] == pytest.approx(1.0)
    assert result["top_k_overlap"] == pytest.approx(1.0)


def test_attribution_map_paints_regions_onto_the_grid():
    label_volume = np.zeros((4, 4, 4), dtype=int)
    label_volume[0] = 1
    label_volume[1] = 2
    weights = np.zeros((4, 1))
    weights[0] = 1.0   # gm::A
    weights[2] = 3.0   # gm::B
    volume = attribution_map(weights, ["gm::A", "csf::A", "gm::B", "csf::B"],
                             label_volume, ["A", "B"])
    assert volume[0].mean() == pytest.approx(0.25)
    assert volume[1].mean() == pytest.approx(0.75)
    assert volume[2].sum() == 0.0


def test_attribution_map_leaves_unattributed_regions_at_zero():
    """A region present in the atlas but absent from the model stays blank."""
    label_volume = np.zeros((3, 3, 3), dtype=int)
    label_volume[0] = 1
    label_volume[1] = 2
    weights = np.zeros((2, 1))
    weights[0] = 1.0
    volume = attribution_map(weights, ["gm::A", "gm::B"], label_volume, ["A", "B"])
    assert volume[0].mean() == pytest.approx(1.0)
    assert volume[1].sum() == 0.0
