"""Tests for federated gradient boosting and its differentially private variant."""

import numpy as np
import pytest

from src.fedboost_v8 import (
    DP_OTHER,
    DP_REGIONS_CSF,
    DP_REGIONS_GM,
    DPBoostConfig,
    FedBoostConfig,
    _bin,
    _merge_sketches,
    _site_sketch,
    _subset,
    _tree_leaves,
    dp_feature_index,
    feature_importance,
    train_dp_fedboost,
    train_fedboost,
)


@pytest.fixture
def cohort():
    """Three classes, two informative features, a biomarker missing for some
    subjects, a feature that only carries signal through being measured, and an
    all-missing column."""
    rng = np.random.default_rng(0)
    n = 240
    y = np.repeat(np.arange(3), n // 3)
    X = rng.normal(0.0, 1.0, size=(n, 5))
    X[:, 0] += 1.5 * y
    X[:, 1] -= 1.0 * y
    X[rng.random(n) < 0.3, 1] = np.nan                  # partly missing biomarker
    X[:, 2] = 1.0                                       # constant when measured ...
    X[y == 0, 2] = np.nan                               # ... missing for class 0
    X[:, 3] = np.nan                                    # never measured
    clients = [np.arange(0, n, 3), np.arange(1, n, 3), np.arange(2, n, 3)]
    return X, y, clients


def _accuracy(model, X, y):
    return float(np.mean(model.predict_proba(X).argmax(1) == y))


# ---- set-up messages ----------------------------------------------------------

def test_site_sketch_skips_features_a_site_never_measured():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    points, counts = _site_sketch(X, 5)
    assert counts.tolist() == [3, 0]
    assert points[0, 0] == 1.0 and points[0, -1] == 3.0
    assert np.isnan(points[1]).all()


def test_merged_edges_ignore_unmeasured_sites_and_columns():
    a = _site_sketch(np.array([[0.0, np.nan], [1.0, np.nan], [2.0, np.nan]]), 8)
    b = _site_sketch(np.array([[np.nan, np.nan], [np.nan, np.nan]]), 8)
    edges = _merge_sketches([a, b], n_bins=4)
    assert len(edges) == 2
    assert edges[1].size == 0                       # never measured anywhere
    assert np.all(np.diff(edges[0]) > 0)
    assert edges[0].max() < 2.0                     # top edge never equals the max


def test_constant_feature_has_no_edges():
    sketch = _site_sketch(np.ones((10, 1)), 8)
    assert _merge_sketches([sketch], n_bins=4)[0].size == 0


def test_bin_routes_missing_values_to_the_last_bin():
    codes = _bin(np.array([[0.5], [np.nan], [5.0]]), [np.array([1.0, 2.0])], n_bins=4)
    assert codes[:, 0].tolist() == [0, 4, 2]


# ---- federated boosting --------------------------------------------------------

def test_federated_boosting_learns_and_reports_communication(cohort):
    X, y, clients = cohort
    model = train_fedboost(X, y, clients, FedBoostConfig(rounds=15, min_samples_leaf=10), 3)
    assert _accuracy(model, X, y) > 0.8
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(1), 1.0)
    assert model.comm["n_sites"] == 3
    assert model.comm["histogram_messages"] > 0
    imp = feature_importance(model, X.shape[1])
    assert imp.sum() == pytest.approx(1.0)
    assert imp[3] == 0.0                            # the never-measured column
    thresholds = np.concatenate([t["threshold"] for rt in model.trees for t in rt])
    assert np.isinf(thresholds).any()               # a measured-vs-missing split


def test_unweighted_boosting_uses_the_class_prior(cohort):
    X, y, clients = cohort
    keep = np.r_[0:80, 80:100, 160:240]
    sub = [np.flatnonzero(np.isin(keep, c)) for c in clients]
    model = train_fedboost(X[keep], y[keep], sub,
                           FedBoostConfig(rounds=1, class_weight="none"), 3)
    prior = np.bincount(y[keep]) / len(keep)
    assert np.allclose(np.exp(model.init), prior)


def test_a_tree_stops_when_no_split_is_allowed(cohort):
    X, y, clients = cohort
    model = train_fedboost(X, y, clients, FedBoostConfig(rounds=1, min_samples_leaf=1000), 3)
    assert all(len(t["feature"]) == 1 for t in model.trees[0])


def test_every_subject_must_belong_to_a_client(cohort):
    X, y, clients = cohort
    with pytest.raises(ValueError):
        train_fedboost(X, y, clients[:2], FedBoostConfig(rounds=1), 3)


# ---- differentially private boosting --------------------------------------------

def test_dp_boosting_without_noise_learns(cohort):
    X, y, clients = cohort
    cfg = DPBoostConfig(epsilon=None, rounds=8, sample_rate=1.0, min_hess=0.5)
    model = train_dp_fedboost(X, y, clients, cfg, 3)
    assert _accuracy(model, X, y) > 0.7
    assert model.comm["epsilon"] is None
    assert model.comm["sigma_histogram"] == 0.0
    thresholds = np.concatenate([t["threshold"] for rt in model.trees for t in rt])
    assert np.isinf(thresholds).any()


def test_dp_boosting_meets_its_budget_and_adds_noise(cohort):
    X, y, clients = cohort
    cfg = DPBoostConfig(epsilon=2.0, rounds=3, colsample=2, seed=1)
    model = train_dp_fedboost(X, y, clients, cfg, 3)
    assert model.comm["epsilon"] == pytest.approx(2.0, rel=0.05)
    assert model.comm["sigma_histogram"] > model.comm["sigma_leaf"] > 0
    for rt in model.trees:
        for t in rt:
            leaves = t["feature"] < 0
            assert np.all(np.abs(t["value"][leaves]) <= cfg.learning_rate + 1e-12)
    assert model.predict_proba(X).shape == (len(y), 3)


def test_tree_leaves_follow_the_missing_direction():
    t = {"feature": np.array([0, -1, -1]), "threshold": np.array([0.0, np.nan, np.nan]),
         "missing_left": np.array([False, True, True]),
         "left": np.array([1, -1, -1]), "right": np.array([2, -1, -1])}
    X = np.array([[-1.0], [1.0], [np.nan]])
    assert _tree_leaves(t, X).tolist() == [1, 2, 2]


def test_dp_feature_index_is_fixed_a_priori():
    names = [f"neuromorphometrics:gm::{s} {r}" for r in DP_REGIONS_GM for s in ("Left", "Right")]
    names += [f"neuromorphometrics:cs::{s} {r}" for r in DP_REGIONS_CSF for s in ("Left", "Right")]
    names = ["other"] + names[::-1] + DP_OTHER
    idx = dp_feature_index(names)
    assert len(idx) == 2 * len(DP_REGIONS_GM) + 2 * len(DP_REGIONS_CSF) + len(DP_OTHER)
    assert names[idx[0]] == f"neuromorphometrics:gm::Left {DP_REGIONS_GM[0]}"
    with pytest.raises(ValueError):
        dp_feature_index(names[:-1])


def test_subset_remaps_labels_to_the_chosen_classes():
    X = np.arange(8.0).reshape(4, 2)
    y = np.array([0, 1, 2, 0])
    Xs, ys, ss, k = _subset(X, y, ["a", "b", "c", "d"], "cn-ad")
    assert k == 2
    assert ys.tolist() == [0, 1, 0]
    assert ss == ["a", "c", "d"]
    assert Xs.shape == (3, 2)


def test_a_tree_stops_at_the_leaf_budget(cohort):
    X, y, clients = cohort
    model = train_fedboost(X, y, clients,
                           FedBoostConfig(rounds=1, max_leaf_nodes=2, min_samples_leaf=5), 3)
    assert all(int(np.sum(t["feature"] < 0)) == 2 for t in model.trees[0])
