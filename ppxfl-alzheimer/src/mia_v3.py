"""
mia_v3.py — subject-level membership inference, as an empirical check on the
formal privacy guarantee.

An epsilon is a worst-case bound over all adversaries and all neighbouring
datasets; it says what *cannot* happen, not what an attacker actually achieves.
Reporting an attack alongside it turns the guarantee into something a reader can
calibrate: a mechanism whose formal epsilon is loose may still leave an attacker
at chance, and a mechanism at a comfortable epsilon that leaves an attacker well
above chance would be worth knowing about.

The attack is the standard loss-threshold one, at the granularity the guarantee
is stated in. Each subject contributes one scan, so a subject's membership
signal is the loss of the trained model on that subject: members were trained
on, non-members were not, and members tend to have lower loss. The attack's
AUROC over that score is the reported quantity, with 0.5 as chance.

No shadow models are trained. A threshold attack understates what a stronger
adversary could do, so the number is a lower bound on leakage and is reported
as one.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .evaluate_v3 import LABEL_SETS, load_features
from .federated_v3 import FedConfig, calibrate_noise, predict_proba, train_federated
from .splits_v3 import make_clients, make_folds


def per_subject_loss(X, y, result) -> np.ndarray:
    """Cross-entropy of the trained model on each subject's own scan."""
    probability = predict_proba(X, result)
    picked = probability[np.arange(len(y)), y]
    return -np.log(np.clip(picked, 1e-12, 1.0))


def attack_auroc(member_scores, non_member_scores) -> float:
    """AUROC of a loss-threshold attack separating members from non-members.

    Lower loss is taken as evidence of membership, so the score fed to the AUROC
    is the negated loss.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.concatenate([
        np.ones(len(member_scores)), np.zeros(len(non_member_scores))
    ])
    scores = -np.concatenate([member_scores, non_member_scores])
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def run(X, y, sites, label_set: str, config: FedConfig, n_splits: int,
        client_scheme: str = "natural") -> dict:
    """Train on each fold and attack the model with its own held-out subjects."""
    classes = LABEL_SETS[label_set]["classes"]
    keep = np.isin(y, classes)
    X_sub, y_sub = X[keep], y[keep]
    sites_sub = [s for s, k in zip(sites, keep) if k]
    remap = {c: i for i, c in enumerate(classes)}
    y_sub = np.asarray([remap[v] for v in y_sub])

    folds = make_folds("subject", y_sub, sites_sub, n_splits=n_splits, seed=config.seed)
    member, non_member = [], []
    for fold in folds:
        mean = X_sub[fold.train].mean(axis=0)
        std = X_sub[fold.train].std(axis=0)
        std[std == 0] = 1.0
        train_X = (X_sub[fold.train] - mean) / std
        test_X = (X_sub[fold.test] - mean) / std

        clients = make_clients(
            client_scheme, np.arange(len(fold.train)), y_sub[fold.train],
            [sites_sub[i] for i in fold.train], seed=config.seed,
        )
        result = train_federated(train_X, y_sub[fold.train], clients, config,
                                 n_classes=len(classes))
        member.append(per_subject_loss(train_X, y_sub[fold.train], result))
        non_member.append(per_subject_loss(test_X, y_sub[fold.test], result))

    member = np.concatenate(member)
    non_member = np.concatenate(non_member)
    return {
        "label_set": label_set,
        "client_scheme": client_scheme,
        "noise_multiplier": config.noise_multiplier,
        "seed": config.seed,
        "n_members": int(len(member)),
        "n_non_members": int(len(non_member)),
        "attack_auroc": attack_auroc(member, non_member),
        "member_loss_mean": float(member.mean()),
        "non_member_loss_mean": float(non_member.mean()),
        "loss_gap": float(non_member.mean() - member.mean()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="subject-level membership inference")
    parser.add_argument("--features", default=os.path.join("data", "mni2mm", "features.npz"))
    parser.add_argument("--out-dir", default=os.path.join("results_v3", "metrics"))
    parser.add_argument("--label-sets", nargs="+", default=["cn-mci-ad"])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--subject-sample-rate", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args(argv)

    X, y, subjects, sites, feature_names = load_features(args.features)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for label_set in args.label_sets:
        for epsilon in [None] + list(args.epsilons):
            noise = 0.0 if epsilon is None else calibrate_noise(
                epsilon, args.subject_sample_rate, args.rounds, args.delta
            )
            for seed in args.seeds:
                config = FedConfig(
                    rounds=args.rounds, local_epochs=args.local_epochs,
                    learning_rate=args.learning_rate, clip_norm=args.clip_norm,
                    noise_multiplier=noise,
                    subject_sample_rate=args.subject_sample_rate,
                    seed=seed, delta=args.delta,
                )
                payload = run(X, y, sites, label_set, config, args.n_splits)
                payload["target_epsilon"] = epsilon
                rows.append(payload)
                label = "nonprivate" if epsilon is None else f"eps{epsilon:g}"
                print(f"  mia {label_set} {label} s{seed}: "
                      f"AUROC={payload['attack_auroc']:.3f} "
                      f"loss gap={payload['loss_gap']:+.3f}", flush=True)

    with open(os.path.join(args.out_dir, "mia_v3.json"), "w", encoding="utf-8") as handle:
        json.dump({"runs": rows}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
