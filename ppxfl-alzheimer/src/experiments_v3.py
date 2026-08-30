"""
experiments_v3.py — the federated, private and explainability experiment matrix.

Three questions are answered here, each against a control measured the same way:

1. *What does federating cost, and does the partition matter?*  The natural
   partition puts one client per real ADNI site, so the heterogeneity is the
   cohort's own — scanner, protocol and a label mix that leaves some sites with
   no MCI subjects and others with no CN.  It is compared against an IID
   partition and a Dirichlet partition of the same subjects, so the effect of
   using the real partition is measured rather than asserted.

2. *What does subject-level differential privacy cost, and what drives it?*
   The same mechanism is run at several model dimensions.  The Gaussian noise
   added to a summed update has expected norm proportional to ``sqrt(d)`` while
   the update itself does not grow with ``d``, so dimension — not the privacy
   budget alone — is what decides whether a private federated model is usable.
   Both the accuracy and the measured noise-to-signal ratio are recorded.

3. *Do private models still look at the right anatomy?*  Because every model
   here is defined over named MNI regions, its attribution can be compared
   directly against the non-private centralised reference and against the
   medial temporal structures where Alzheimer's atrophy is best established.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .evaluate_v3 import LABEL_SETS, load_features
from .federated_v3 import (
    FedConfig,
    calibrate_noise,
    dump_result,
    noise_to_signal_law,
    predict_proba,
    summarise_trace,
    train_federated,
)
from .splits_v3 import make_clients, make_folds, partition_summary
from .xai_v3 import compare

# Model dimensions the privacy analysis contrasts.  The first two are the
# configurations the previous version of this study reported; the third is the
# representation introduced here.
REFERENCE_DIMENSIONS = {
    "resnet50_full": 23_508_035,
    "resnet50_head": 6_147,
}


def _metrics(y_true, y_pred, y_score, n_classes):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        if n_classes == 2:
            out["macro_auroc"] = float(roc_auc_score(y_true, y_score[:, 1]))
        else:
            out["macro_auroc"] = float(
                roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
            )
    except ValueError:
        out["macro_auroc"] = float("nan")
    return out


def _standardise(train_X, test_X):
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0
    return (train_X - mean) / std, (test_X - mean) / std


def run_federated_configuration(X, y, sites, label_set: str, split_scheme: str,
                                client_scheme: str, config: FedConfig,
                                n_splits: int, feature_names) -> dict:
    """Cross-validated federated training for one configuration."""
    classes = LABEL_SETS[label_set]["classes"]
    keep = np.isin(y, classes)
    X_sub, y_sub = X[keep], y[keep]
    sites_sub = [s for s, k in zip(sites, keep) if k]
    remap = {c: i for i, c in enumerate(classes)}
    y_sub = np.asarray([remap[v] for v in y_sub])
    n_classes = len(classes)

    folds = make_folds(split_scheme, y_sub, sites_sub, n_splits=n_splits,
                       seed=config.seed)
    predictions = np.zeros(len(y_sub), dtype=int)
    scores = np.zeros((len(y_sub), n_classes), dtype=float)
    traces, partitions, weight_sets = [], [], []

    for fold in folds:
        train_X, test_X = _standardise(X_sub[fold.train], X_sub[fold.test])
        train_y = y_sub[fold.train]
        train_sites = [sites_sub[i] for i in fold.train]

        local_indices = np.arange(len(fold.train))
        clients = make_clients(
            client_scheme, local_indices, train_y, train_sites,
            n_clients=8, seed=config.seed
        )
        partitions.append(partition_summary(clients, train_y, train_sites))

        result = train_federated(train_X, train_y, clients, config, n_classes=n_classes)
        weight_sets.append(result.weights)
        traces.append(summarise_trace(result.trace))

        probability = predict_proba(test_X, result)
        scores[fold.test] = probability
        predictions[fold.test] = np.argmax(probability, axis=1)

    overall = _metrics(y_sub, predictions, scores, n_classes)
    counts = np.bincount(y_sub, minlength=n_classes)
    dimension = X_sub.shape[1] * n_classes + n_classes

    payload = {
        "label_set": label_set,
        "split_scheme": split_scheme,
        "client_scheme": client_scheme,
        "n_subjects": int(len(y_sub)),
        "chance_accuracy": float(counts.max() / counts.sum()),
        "uniform_chance": 1.0 / n_classes,
        "overall": overall,
        "perturbed_dimension": int(dimension),
        "noise_multiplier": config.noise_multiplier,
        "clip_norm": config.clip_norm,
        "rounds": config.rounds,
        "local_epochs": config.local_epochs,
        "subject_sample_rate": config.subject_sample_rate,
        "delta": config.delta,
        "seed": config.seed,
        "trace": traces,
        "partition": partitions[0] if partitions else {},
        "mean_weights": np.mean(weight_sets, axis=0).tolist() if weight_sets else [],
    }
    if config.noise_multiplier > 0:
        from .federated_v3 import compute_epsilon

        payload["epsilon"] = compute_epsilon(
            config.noise_multiplier, config.subject_sample_rate,
            config.rounds, config.delta
        )
    return payload


def command_federated(args) -> None:
    X, y, subjects, sites, feature_names = load_features(args.features)
    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for label_set in args.label_sets:
        for split_scheme in args.schemes:
            for client_scheme in args.client_schemes:
                for seed in args.seeds:
                    config = FedConfig(
                        rounds=args.rounds, local_epochs=args.local_epochs,
                        learning_rate=args.learning_rate,
                        clip_norm=args.clip_norm, seed=seed
                    )
                    payload = run_federated_configuration(
                        X, y, sites, label_set, split_scheme, client_scheme,
                        config, args.n_splits, feature_names
                    )
                    tag = f"fed_{client_scheme}_{label_set}_{split_scheme}_s{seed}"
                    dump_result(os.path.join(args.out_dir, f"{tag}.json"), payload)
                    summary.append({"tag": tag, **payload["overall"]})
                    print(f"  {tag}: acc={payload['overall']['accuracy']:.3f} "
                          f"bal={payload['overall']['balanced_accuracy']:.3f} "
                          f"F1={payload['overall']['macro_f1']:.3f}", flush=True)
    dump_result(os.path.join(args.out_dir, "federated_summary.json"), {"runs": summary})


def command_privacy(args) -> None:
    X, y, subjects, sites, feature_names = load_features(args.features)
    os.makedirs(args.out_dir, exist_ok=True)

    # One reference per label set. A three-class model and a two-class model
    # are different decision problems, so comparing a private CN-vs-AD model's
    # attribution against a non-private CN/MCI/AD reference would report a
    # difference of task as a cost of privacy.
    references: dict[str, np.ndarray] = {}
    summary = []
    for label_set in args.label_sets:
        for epsilon_target in [None] + list(args.epsilons):
            for seed in args.seeds:
                if epsilon_target is None:
                    noise = 0.0
                else:
                    noise = calibrate_noise(
                        epsilon_target, args.subject_sample_rate,
                        args.rounds, args.delta
                    )
                config = FedConfig(
                    rounds=args.rounds, local_epochs=args.local_epochs,
                    learning_rate=args.learning_rate,
                    clip_norm=args.clip_norm, noise_multiplier=noise,
                    subject_sample_rate=args.subject_sample_rate,
                    seed=seed, delta=args.delta
                )
                payload = run_federated_configuration(
                    X, y, sites, label_set, args.scheme, args.client_scheme,
                    config, args.n_splits, feature_names
                )
                payload["target_epsilon"] = epsilon_target

                weights = np.asarray(payload["mean_weights"])
                if epsilon_target is None and label_set not in references:
                    references[label_set] = weights
                reference = references.get(label_set)
                if reference is not None and weights.size:
                    payload["explainability"] = compare(
                        reference, weights, feature_names, k=args.top_k
                    )

                # The same mechanism applied to the model dimensions the
                # previous study used, so the cost of dimension is explicit.
                participants = int(payload["n_subjects"] * (args.n_splits - 1)
                                   / args.n_splits * args.subject_sample_rate)
                payload["dimension_comparison"] = [
                    noise_to_signal_law(dimension, noise, args.clip_norm, participants)
                    for dimension in (
                        [payload["perturbed_dimension"]]
                        + list(REFERENCE_DIMENSIONS.values())
                    )
                ] if noise > 0 else []

                label = "nonprivate" if epsilon_target is None else f"eps{epsilon_target}"
                tag = f"dp_{label_set}_{label}_s{seed}"
                dump_result(os.path.join(args.out_dir, f"{tag}.json"), payload)
                summary.append({
                    "tag": tag,
                    "label_set": label_set,
                    "target_epsilon": epsilon_target,
                    "accounted_epsilon": payload.get("epsilon"),
                    "noise_multiplier": noise,
                    **payload["overall"],
                    "explainability": payload.get("explainability", {}).get(
                        "agreement_spearman"
                    ),
                })
                print(f"  {tag}: eps={epsilon_target} sigma={noise:.2f} "
                      f"acc={payload['overall']['accuracy']:.3f} "
                      f"F1={payload['overall']['macro_f1']:.3f}", flush=True)

    dump_result(os.path.join(args.out_dir, "privacy_summary.json"), {"runs": summary})


def command_dimension_law(args) -> None:
    """The closed-form noise-to-signal ratio across model dimensions."""
    X, y, _, _, feature_names = load_features(args.features)
    dimensions = {
        "roi_logreg": X.shape[1] * 3 + 3,
        **REFERENCE_DIMENSIONS,
    }
    participants = int(X.shape[0] * (args.n_splits - 1) / args.n_splits)
    rows = []
    for epsilon in args.epsilons:
        noise = calibrate_noise(epsilon, args.subject_sample_rate, args.rounds,
                                args.delta)
        for name, dimension in dimensions.items():
            law = noise_to_signal_law(dimension, noise, args.clip_norm, participants)
            rows.append({"model": name, "target_epsilon": epsilon,
                         "noise_multiplier": noise, **law})
            print(f"  eps={epsilon} {name}: d={dimension:,} "
                  f"noise/signal={law['worst_case_ratio']:.3f}")
    dump_result(os.path.join(args.out_dir, "dimension_law.json"),
                {"participants": participants, "rows": rows})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="v3 federated / privacy experiments")
    parser.add_argument("--features", default=os.path.join("data", "mni2mm", "features.npz"))
    parser.add_argument("--out-dir", default=os.path.join("results_v3", "metrics"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--subject-sample-rate", type=float, default=0.5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--label-sets", nargs="+", default=["cn-mci-ad", "cn-ad"])
    parser.add_argument("--top-k", type=int, default=10)

    subparsers = parser.add_subparsers(dest="command", required=True)

    federated = subparsers.add_parser("federated")
    federated.add_argument("--schemes", nargs="+", default=["subject", "site"])
    federated.add_argument("--client-schemes", nargs="+",
                           default=["natural", "iid", "dirichlet"])
    federated.set_defaults(func=command_federated)

    privacy = subparsers.add_parser("privacy")
    privacy.add_argument("--scheme", default="subject")
    privacy.add_argument("--client-scheme", default="natural")
    privacy.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 2.0, 5.0, 10.0])
    privacy.set_defaults(func=command_privacy)

    law = subparsers.add_parser("dimension-law")
    law.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 2.0, 5.0, 10.0])
    law.set_defaults(func=command_dimension_law)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
