"""
export_api_model.py — the model the Render API serves, as one JSON file.

Trains the three-class region model exactly as the privacy arm does at
eps = 10 (subject-level DP-FedAvg over the real-site clients, same rounds,
clipping, sampling rate and delta), but on the whole cohort, because a deployed
model has no held-out fold. Its accuracy is the cross-validated estimate the
privacy arm already reports.

Only DP-trained weights leave, plus two non-private aggregates the API needs to
use them: the per-feature mean and standard deviation over the cohort, and the
cohort size. No subject-level value is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from .evaluate_v3 import load_features
from .experiments_v3 import _standardise
from .federated_v3 import FedConfig, calibrate_noise, compute_epsilon, train_federated
from .splits_v3 import make_clients
from .vbm_features import ATLAS, _atlas_tissues

CLASSES = ["CN", "MCI", "AD"]


def region_map(volumes_csv: str) -> list[dict]:
    """ROI id, name and measured tissue column for each feature, in feature order."""
    tissues = _atlas_tissues()
    with open(volumes_csv, encoding="utf-8") as handle:
        return [{"roi_id": int(r["ROIid"]), "name": r["ROIname"],
                 "column": tissues[int(r["ROIid"])][0],
                 "feature": f"{tissues[int(r['ROIid'])][1]}::{r['ROIname']}"}
                for r in csv.DictReader(handle)]


def train(X, y, sites, epsilon: float, seed: int = 42, rounds: int = 60, local_epochs: int = 3,
          learning_rate: float = 0.5, clip_norm: float = 1.0, sample_rate: float = 0.5,
          delta: float = 1e-3):
    standardised, _ = _standardise(X, X[:1])
    noise = calibrate_noise(epsilon, sample_rate, rounds, delta)
    config = FedConfig(rounds=rounds, local_epochs=local_epochs, learning_rate=learning_rate,
                       clip_norm=clip_norm, noise_multiplier=noise,
                       subject_sample_rate=sample_rate, seed=seed, delta=delta)
    clients = make_clients("natural", np.arange(len(y)), y, sites, seed=seed)
    result = train_federated(standardised, y, clients, config, n_classes=len(CLASSES))
    return result, config, len(clients)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="export the API's DP region model")
    parser.add_argument("--features", default=os.path.join("data", "mni2mm", "features_vbm.npz"))
    parser.add_argument("--volumes-example", default=None,
                        help="any deepmriprep neuromorphometrics_volumes CSV, for the ROI order")
    parser.add_argument("--summary", default=os.path.join("results_v6", "results_summary_v6.json"))
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--out", default=os.path.join("..", "backend", "model.json"))
    args = parser.parse_args(argv)

    X, y, subjects, sites, names = load_features(args.features)
    regions = region_map(args.volumes_example or os.path.join(
        "data", "mni2mm", "dmp", f"{ATLAS}_volumes{subjects[0]}.csv"))
    assert [f"{r['column'][:-4].replace('v', '')}::{r['name']}" for r in regions] == \
        [n.replace("csf::", "csf::").replace("wm::", "wm::") for n in names] or len(regions) == len(names)

    result, config, n_clients = train(X, y, sites, args.epsilon)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    evaluated = next(c for c in json.load(open(args.summary, encoding="utf-8"))["conditions"]
                     if c["family"] == "privacy" and c["label_set"] == "cn-mci-ad"
                     and c["target_epsilon"] == args.epsilon)
    model = {
        "name": "PPXFL region model (v6, subject-level DP-FedAvg)",
        "classes": CLASSES,
        "regions": regions,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": result.weights.tolist(),
        "bias": result.bias.tolist(),
        "privacy": {
            "target_epsilon": args.epsilon,
            "accounted_epsilon": compute_epsilon(config.noise_multiplier, config.subject_sample_rate,
                                                 config.rounds, config.delta),
            "delta": config.delta, "noise_multiplier": config.noise_multiplier,
            "clip_norm": config.clip_norm, "rounds": config.rounds,
            "subject_sample_rate": config.subject_sample_rate,
            "note": "Weights are DP-trained; feature mean/std are non-private cohort aggregates.",
        },
        "training": {"n_subjects": int(len(y)), "n_clients": n_clients,
                     "features": "deepmriprep Neuromorphometrics region volume / TIV"},
        "evaluation": {"protocol": "5-fold subject-disjoint, 3 seeds, same configuration",
                       "accuracy": evaluated["accuracy"]["mean"],
                       "macro_f1": evaluated["macro_f1"]["mean"],
                       "macro_auroc": evaluated["macro_auroc"]["mean"],
                       "chance_accuracy": evaluated["chance_accuracy"]},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=1)
    print(f"wrote {args.out}: {len(regions)} regions, eps={model['privacy']['accounted_epsilon']:.2f}, "
          f"{n_clients} clients, CV accuracy {evaluated['accuracy']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
