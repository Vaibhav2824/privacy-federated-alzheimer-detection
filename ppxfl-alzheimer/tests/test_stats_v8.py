"""Tests for the paired significance tests in stats_v8 (synthetic data only)."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.stats_v8 import (
    _midrank,
    compare,
    delong,
    macro_auroc,
    mcnemar,
    paired_bootstrap,
    wilson,
)


def test_mcnemar_counts_discordant_pairs_and_corrects_for_continuity():
    a = np.array([1, 1, 1, 0, 0, 1, 0, 1], bool)
    b = np.array([1, 0, 0, 0, 1, 1, 0, 0], bool)
    out = mcnemar(a, b)
    assert (out["b"], out["c"]) == (3, 1)
    assert out["statistic"] == pytest.approx((abs(3 - 1) - 1) ** 2 / 4)
    assert 0 < out["p"] < 1


def test_mcnemar_with_no_discordant_pairs_is_not_significant():
    same = np.array([1, 0, 1], bool)
    assert mcnemar(same, same) == {"b": 0, "c": 0, "statistic": 0.0, "p": 1.0}


def test_wilson_interval_contains_the_proportion():
    lo, hi = wilson(45, 50)
    assert lo < 0.9 < hi and 0 <= lo and hi <= 1
    assert np.isnan(wilson(0, 0)[0])


def test_midrank_averages_ties():
    assert _midrank(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_delong_matches_sklearn_auc_and_detects_a_real_difference():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 150)
    good = y + rng.normal(0, 0.6, len(y))
    weak = y + rng.normal(0, 2.0, len(y))
    out = delong(y, good, weak)
    assert out["auc_a"] == pytest.approx(roc_auc_score(y, good))
    assert out["auc_b"] == pytest.approx(roc_auc_score(y, weak))
    assert out["p"] < 0.01


def test_delong_on_identical_scores_is_not_significant():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    assert delong(y, s, s)["p"] == 1.0


def _scores(y, n_classes, noise, seed):
    rng = np.random.default_rng(seed)
    logits = np.eye(n_classes)[y] * 2 + rng.normal(0, noise, (len(y), n_classes))
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def test_macro_auroc_handles_two_and_three_classes():
    y2 = np.repeat([0, 1], 20)
    s2 = _scores(y2, 2, 1.0, 0)
    assert macro_auroc(y2, s2) == pytest.approx(roc_auc_score(y2, s2[:, 1]))
    y3 = np.repeat([0, 1, 2], 20)
    assert 0.5 < macro_auroc(y3, _scores(y3, 3, 1.0, 1)) <= 1


def test_paired_bootstrap_skips_resamples_that_miss_a_class():
    y = np.array([0, 1, 2])
    s = _scores(y, 3, 0.1, 2)
    out = paired_bootstrap(y, s, s, n_boot=50, seed=0)
    assert out["n_resamples"] < 50
    assert out["acc_diff_ci"] == [0.0, 0.0]


def test_compare_uses_delong_for_two_classes_and_bootstrap_for_three():
    y2 = np.repeat([0, 1], 60)
    two = compare(y2, _scores(y2, 2, 0.8, 3), _scores(y2, 2, 1.5, 4), seed_check=True)
    assert two["auroc_test"]["method"] == "DeLong"
    assert "delong_vs_bootstrap_p" in two
    assert two["acc_diff"] == pytest.approx(two["acc_a"] - two["acc_b"])
    y3 = np.repeat([0, 1, 2], 40)
    three = compare(y3, _scores(y3, 3, 0.8, 5), _scores(y3, 3, 0.8, 6))
    assert three["auroc_test"]["method"] == "paired bootstrap"
    assert "delong_vs_bootstrap_p" not in three


def test_compare_without_the_bootstrap_check():
    y2 = np.repeat([0, 1], 40)
    out = compare(y2, _scores(y2, 2, 0.8, 7), _scores(y2, 2, 0.8, 8))
    assert out["auroc_test"]["method"] == "DeLong" and "delong_vs_bootstrap_p" not in out
