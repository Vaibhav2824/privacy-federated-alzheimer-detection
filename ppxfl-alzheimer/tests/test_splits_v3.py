"""Tests for the v3 evaluation splits and federated client partitions."""

import numpy as np
import pytest

from src.splits_v3 import (
    dirichlet_clients,
    iid_clients,
    make_clients,
    make_folds,
    natural_clients,
    partition_summary,
    site_folds,
    subject_folds,
)


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.zeros(40), np.ones(40), np.full(40, 2)]).astype(int)
    sites = [f"{(i % 12) + 1:03d}" for i in range(120)]
    rng.shuffle(labels)
    return labels, sites


def test_subject_folds_cover_every_subject_exactly_once(cohort):
    labels, _ = cohort
    folds = subject_folds(labels, n_splits=5, seed=42)
    assert len(folds) == 5
    test_indices = np.concatenate([f.test for f in folds])
    assert sorted(test_indices) == list(range(len(labels)))


def test_subject_folds_train_and_test_are_disjoint(cohort):
    labels, _ = cohort
    for fold in subject_folds(labels, n_splits=5, seed=42):
        assert not set(fold.train) & set(fold.test)
        assert fold.scheme == "subject"


def test_site_folds_hold_out_whole_sites(cohort):
    """A site that appears in a test fold must not appear in its training set."""
    labels, sites = cohort
    sites = np.asarray(sites)
    for fold in site_folds(labels, sites, n_splits=4, seed=42):
        assert not set(sites[fold.test]) & set(sites[fold.train])


def test_make_folds_rejects_unknown_scheme(cohort):
    labels, sites = cohort
    with pytest.raises(ValueError, match="unknown split scheme"):
        make_folds("nonsense", labels, sites)


def test_natural_clients_pool_small_sites():
    sites = np.asarray(["001"] * 10 + ["002"] * 9 + ["003"] * 2 + ["004"] * 1)
    clients = natural_clients(np.arange(len(sites)), sites, min_subjects=8)
    sizes = sorted(len(c) for c in clients)
    # 001 and 002 stand alone; 003 and 004 are pooled into one client of three.
    assert sizes == [3, 9, 10]


def test_natural_clients_keep_every_subject():
    sites = np.asarray(["001"] * 10 + ["009"] * 3)
    clients = natural_clients(np.arange(len(sites)), sites, min_subjects=8)
    assert sorted(np.concatenate(clients)) == list(range(len(sites)))


def test_natural_clients_without_small_sites_have_no_pool():
    sites = np.asarray(["001"] * 10 + ["002"] * 10)
    clients = natural_clients(np.arange(20), sites, min_subjects=8)
    assert len(clients) == 2


def test_iid_clients_partition_the_indices():
    clients = iid_clients(np.arange(30), 4, seed=1)
    assert len(clients) == 4
    assert sorted(np.concatenate(clients)) == list(range(30))


def test_dirichlet_clients_partition_the_indices(cohort):
    labels, _ = cohort
    clients = dirichlet_clients(np.arange(len(labels)), labels, 5, alpha=0.3, seed=7)
    assert len(clients) == 5
    assert sorted(np.concatenate(clients)) == list(range(len(labels)))


def test_dirichlet_is_more_skewed_than_iid(cohort):
    """The simulated non-IID partition should have lower mean label entropy."""
    labels, sites = cohort
    indices = np.arange(len(labels))
    iid = partition_summary(iid_clients(indices, 5, seed=3), labels, sites)
    skewed = partition_summary(
        dirichlet_clients(indices, labels, 5, alpha=0.1, seed=3), labels, sites
    )
    iid_entropy = np.mean([c["label_entropy"] for c in iid["clients"]])
    skewed_entropy = np.mean([c["label_entropy"] for c in skewed["clients"]])
    assert skewed_entropy < iid_entropy


def test_make_clients_dispatches_each_scheme(cohort):
    labels, sites = cohort
    indices = np.arange(len(labels))
    for scheme in ("natural", "iid", "dirichlet"):
        clients = make_clients(scheme, indices, labels, sites, n_clients=4, seed=0)
        assert sorted(np.concatenate(clients)) == list(range(len(labels)))


def test_make_clients_rejects_unknown_scheme(cohort):
    labels, sites = cohort
    with pytest.raises(ValueError, match="unknown client scheme"):
        make_clients("nonsense", np.arange(len(labels)), labels, sites)


def test_partition_summary_reports_missing_classes():
    labels = np.array([0, 0, 1, 1, 2, 2])
    sites = ["001"] * 6
    clients = [np.array([0, 1]), np.array([2, 3, 4, 5])]
    summary = partition_summary(clients, labels, sites)
    assert summary["n_clients"] == 2
    assert summary["clients"][0]["missing_classes"] == [1, 2]
    assert summary["clients"][1]["missing_classes"] == [0]
    assert summary["clients_missing_a_class"] == 2


def test_partition_summary_entropy_is_zero_for_single_class():
    labels = np.array([1, 1, 1])
    summary = partition_summary([np.array([0, 1, 2])], labels, ["001"] * 3)
    assert summary["clients"][0]["label_entropy"] == pytest.approx(0.0)
