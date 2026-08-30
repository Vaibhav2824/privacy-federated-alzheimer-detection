"""
splits_v3.py — evaluation splits and federated client partitions.

Two split schemes are provided, because they answer different questions:

``subject``
    Stratified K-fold over subjects.  One scan per subject means subject
    disjointness is structural rather than something to be checked afterwards.
    This is the standard protocol and the one comparable to the literature.

``site``
    Stratified group K-fold with the ADNI site as the group, so every test site
    is unseen during training.  A federated model is deployed at hospitals that
    did not contribute to it, and scanner and protocol vary by site, so this is
    the number that reflects deployment.  It is expected to be lower than the
    subject-level number; reporting both is the point.

The federated partitions are similarly paired.  ``natural`` gives one client
per real ADNI site: the label skew is whatever the sites actually have — in
this cohort one site contributes no MCI at all and another no CN — and the
feature skew is real scanner and protocol variation rather than a Dirichlet
draw.  ``iid`` and ``dirichlet`` are the usual simulated alternatives, kept so
that the effect of using the real partition can be measured instead of
asserted.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray
    test: np.ndarray
    scheme: str


def subject_folds(labels, n_splits: int = 5, seed: int = 42) -> list[Fold]:
    """Stratified K-fold over subjects."""
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        Fold(index=i, train=train, test=test, scheme="subject")
        for i, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels))
    ]


def site_folds(labels, sites, n_splits: int = 5, seed: int = 42) -> list[Fold]:
    """Stratified group K-fold holding out whole ADNI sites."""
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray(labels)
    sites = np.asarray(sites)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        Fold(index=i, train=train, test=test, scheme="site")
        for i, (train, test) in enumerate(
            splitter.split(np.zeros(len(labels)), labels, groups=sites)
        )
    ]


def make_folds(scheme: str, labels, sites, n_splits: int = 5, seed: int = 42) -> list[Fold]:
    if scheme == "subject":
        return subject_folds(labels, n_splits=n_splits, seed=seed)
    if scheme == "site":
        return site_folds(labels, sites, n_splits=n_splits, seed=seed)
    raise ValueError(f"unknown split scheme: {scheme}")


def natural_clients(indices, sites, min_subjects: int = 8) -> list[np.ndarray]:
    """One client per ADNI site; sites below ``min_subjects`` are pooled.

    Pooling rather than dropping keeps every subject in training.  The pooled
    client stands for the small contributing centres that a real federation
    would also have to carry, and it is reported separately in the partition
    summary so its effect is visible.
    """
    indices = np.asarray(indices)
    sites = np.asarray(sites)
    by_site: dict[str, list[int]] = collections.defaultdict(list)
    for position in indices:
        by_site[sites[position]].append(int(position))

    clients, pooled = [], []
    for site in sorted(by_site):
        members = by_site[site]
        if len(members) >= min_subjects:
            clients.append(np.asarray(members, dtype=int))
        else:
            pooled.extend(members)
    if pooled:
        clients.append(np.asarray(pooled, dtype=int))
    return clients


def iid_clients(indices, n_clients: int, seed: int = 42) -> list[np.ndarray]:
    """Random equal-sized clients — the homogeneous control."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.asarray(indices))
    return [np.asarray(part, dtype=int) for part in np.array_split(shuffled, n_clients)]


def dirichlet_clients(indices, labels, n_clients: int, alpha: float = 0.5,
                      seed: int = 42) -> list[np.ndarray]:
    """The usual simulated label-skew partition, for comparison with ``natural``."""
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    buckets: list[list[int]] = [[] for _ in range(n_clients)]
    for klass in np.unique(labels[indices]):
        members = rng.permutation(indices[labels[indices] == klass])
        proportions = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(proportions) * len(members)).astype(int)[:-1]
        for client, part in enumerate(np.split(members, cuts)):
            buckets[client].extend(int(i) for i in part)
    return [np.asarray(sorted(b), dtype=int) for b in buckets]


def make_clients(scheme: str, indices, labels, sites, n_clients: int = 8,
                 alpha: float = 0.5, seed: int = 42,
                 min_subjects: int = 8) -> list[np.ndarray]:
    if scheme == "natural":
        return natural_clients(indices, sites, min_subjects=min_subjects)
    if scheme == "iid":
        return iid_clients(indices, n_clients, seed=seed)
    if scheme == "dirichlet":
        return dirichlet_clients(indices, labels, n_clients, alpha=alpha, seed=seed)
    raise ValueError(f"unknown client scheme: {scheme}")


def partition_summary(clients, labels, sites) -> dict:
    """Client sizes, label mix and label skew, for reporting the partition."""
    labels = np.asarray(labels)
    sites = np.asarray(sites)
    rows = []
    for client_index, members in enumerate(clients):
        counts = collections.Counter(int(labels[i]) for i in members)
        total = max(len(members), 1)
        distribution = np.asarray([counts.get(k, 0) / total for k in range(3)])
        rows.append({
            "client": client_index,
            "n_subjects": int(len(members)),
            "sites": sorted({str(sites[i]) for i in members}),
            "class_counts": {str(k): int(counts.get(k, 0)) for k in range(3)},
            "missing_classes": [k for k in range(3) if counts.get(k, 0) == 0],
            "label_entropy": float(
                -np.sum(distribution[distribution > 0] * np.log(distribution[distribution > 0]))
            ),
        })
    return {
        "n_clients": len(clients),
        "clients": rows,
        "clients_missing_a_class": sum(1 for r in rows if r["missing_classes"]),
    }
