"""
export_api_model_v8.py -- the v8 private model the Render API serves, as one JSON file.

Trains the 27-feature, 84-parameter three-class model exactly as the privacy arm
does (subject-level DP-FedAvg over the real-site clients, same rounds, local
epochs, clipping, sampling rate, delta and PRV accountant) but on the whole v8
cohort, because a deployed model has no held-out fold. Its accuracy is the
cross-validated estimate the privacy arm already reports.

Only DP-trained weights leave, plus the non-private cohort aggregates the API
needs to use them, the same kind the v7 export shipped: per-feature mean, SD
and median, and the mean and SD of each composite part. No subject-level value
is written.

    python -m src.export_api_model_v8 --epsilon 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings

import numpy as np

from .evaluate_v3 import load_features
from .experiments_v3 import configure_composites
from .federated_v3 import FedConfig, calibrate_noise, compute_epsilon, train_federated
from .splits_v3 import make_clients

CLASSES = ["CN", "MCI", "AD"]
TISSUE_COLUMN = {"gm": "gmv_mm3", "cs": "csfv_mm3", "wm": "wmv_mm3"}

# Inputs a visitor may add next to the scan. ``feature`` is the model column the
# value feeds; ``transform`` is applied before the composite's z-score.
CLINICAL = [
    {"key": "age", "label": "Age", "unit": "years", "min": 40, "max": 100, "feature": "demo::age"},
    {"key": "sex", "label": "Sex", "unit": "1 = male, 2 = female", "min": 1, "max": 2,
     "feature": "demo::sex"},
    {"key": "education", "label": "Education", "unit": "years", "min": 0, "max": 30,
     "feature": "bio::education"},
    {"key": "apoe4", "label": "APOE ε4 alleles", "unit": "0, 1 or 2", "min": 0, "max": 2,
     "feature": "bio::apoe4"},
    {"key": "amyloid_cl", "label": "Amyloid PET", "unit": "centiloids", "min": -100, "max": 250,
     "feature": "bio::amyloid_cl"},
    {"key": "csf_ptau_ab42", "label": "CSF p-tau181 / Aβ42", "unit": "ratio", "min": 0, "max": 1,
     "feature": "comp::tau::csf_ptau_ab42", "transform": "none"},
    {"key": "plasma_ptau217", "label": "Plasma p-tau217", "unit": "pg/mL", "min": 0.001, "max": 50,
     "feature": "comp::tau::log_plasma_ptau217", "transform": "log"},
    {"key": "plasma_nfl", "label": "Plasma NfL", "unit": "pg/mL", "min": 0.1, "max": 1000,
     "feature": "comp::neuro::log_plasma_nfl", "transform": "log"},
    {"key": "plasma_gfap", "label": "Plasma GFAP", "unit": "pg/mL", "min": 1, "max": 5000,
     "feature": "comp::neuro::log_plasma_gfap", "transform": "log"},
]
LABELS = {"bio::tau_composite": "Tau composite (CSF / plasma p-tau)",
          "bio::neuro_composite": "Neurodegeneration composite (NfL, GFAP)"}


def regions(feature_names, volumes_csv):
    """ROI id and tissue column for every region feature, from a deepmriprep CSV."""
    with open(volumes_csv, encoding="utf-8") as handle:
        roi = {r["ROIname"]: int(r["ROIid"]) for r in csv.DictReader(handle)}
    out = []
    for n in feature_names:
        if not n.startswith("neuromorphometrics:"):
            continue
        tissue, name = n.split(":", 1)[1].split("::")
        out.append({"roi_id": roi[name], "name": name, "column": TISSUE_COLUMN[tissue], "feature": n})
    return out


def model_inputs(X, names):
    """Composite parts z-scored with cohort statistics, then the model feature matrix."""
    model_names = configure_composites(names)
    groups: dict = {}
    for i, n in enumerate(names):
        if n.startswith("comp::"):
            groups.setdefault(n.split("::")[1], []).append(i)
    base = [i for i, n in enumerate(names) if not n.startswith("comp::")]
    columns, composites = [X[:, base]], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for group, idx in groups.items():
            mu, sd = np.nanmean(X[:, idx], axis=0), np.nanstd(X[:, idx], axis=0)
            sd[~(sd > 0)] = 1.0
            columns.append(np.nanmean((X[:, idx] - mu) / sd, axis=1)[:, None])
            composites.append({"name": f"bio::{group}_composite", "parts": [
                {"feature": names[i], "mean": float(m), "std": float(s)}
                for i, m, s in zip(idx, mu, sd)]})
    Z = np.hstack(columns)
    median = np.nanmedian(Z, axis=0)
    filled = np.where(np.isnan(Z), median, Z)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    std[std == 0] = 1.0
    return model_names, [names[i] for i in base], composites, median, mean, std, (filled - mean) / std


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="export the API's v8 DP model")
    parser.add_argument("--features", default=os.path.join("data", "v8", "features_v8_dp27.npz"))
    parser.add_argument("--volumes-example", default=os.path.join(
        "..", "ui", "public", "examples", "mni152_neuromorphometrics_volumes.csv"))
    parser.add_argument("--summary", default=os.path.join("results_v8_summary", "privacy_dp27.json"))
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join("..", "backend", "model.json"))
    args = parser.parse_args(argv)

    X, y, _, sites, names = load_features(args.features)
    model_names, base_names, composites, median, mean, std, Z = model_inputs(X, list(names))

    rounds, sample_rate, delta = 60, 0.5, 1e-3
    noise = calibrate_noise(args.epsilon, sample_rate, rounds, delta, accountant="prv")
    config = FedConfig(rounds=rounds, local_epochs=3, learning_rate=0.5, clip_norm=1.0,
                       noise_multiplier=noise, subject_sample_rate=sample_rate,
                       seed=args.seed, delta=delta, accountant="prv")
    clients = make_clients("natural", np.arange(len(y)), y, sites, seed=args.seed)
    result = train_federated(Z, y, clients, config, n_classes=len(CLASSES))

    with open(args.summary, encoding="utf-8") as handle:
        conditions = json.load(handle)["conditions"]
    evaluated, cn_ad = (next(c for c in conditions if c["family"] == "privacy"
                             and c["label_set"] == ls and c["target_epsilon"] == args.epsilon)
                        for ls in ("cn-mci-ad", "cn-ad"))

    def label(n):
        if n in LABELS:
            return LABELS[n]
        item = next((c for c in CLINICAL if c["feature"] == n), None)
        return item["label"] if item else n.split("::")[-1]

    model = {
        "name": "PPXFL v8 private model (27 features, subject-level DP-FedAvg)",
        "version": 8,
        "classes": CLASSES,
        "regions": regions(base_names, args.volumes_example),
        "clinical": CLINICAL,
        "base_features": base_names,
        "composites": composites,
        "features": [{"feature": n, "label": label(n)} for n in model_names],
        "median": median.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": result.weights.tolist(),
        "bias": result.bias.tolist(),
        "privacy": {
            "target_epsilon": args.epsilon,
            "accounted_epsilon": compute_epsilon(config.noise_multiplier, config.subject_sample_rate,
                                                 config.rounds, config.delta, config.accountant),
            "accountant": config.accountant,
            "delta": config.delta, "noise_multiplier": config.noise_multiplier,
            "clip_norm": config.clip_norm, "rounds": config.rounds,
            "subject_sample_rate": config.subject_sample_rate,
            "parameters": int(result.weights.size + result.bias.size),
            "note": "Weights are DP-trained; feature mean/SD/median and composite-part "
                    "mean/SD are non-private cohort aggregates.",
        },
        "training": {"n_subjects": int(len(y)), "n_clients": len(clients),
                     "features": "20 AD-signature region volumes / TIV (deepmriprep), APOE4, "
                                 "amyloid PET, tau and neurodegeneration composites, "
                                 "education, age, sex; missing values -> cohort median"},
        "evaluation": {"protocol": "5-fold subject-disjoint, 3 seeds, same configuration",
                       "accuracy": evaluated["accuracy"]["mean"],
                       "macro_f1": evaluated["macro_f1"]["mean"],
                       "macro_auroc": evaluated["macro_auroc"]["mean"],
                       "chance_accuracy": evaluated["chance_accuracy"],
                       "cn_ad_accuracy": cn_ad["accuracy"]["mean"]},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=1)
    print(f"wrote {args.out}: {len(model['features'])} features, "
          f"{model['privacy']['parameters']} parameters, "
          f"eps={model['privacy']['accounted_epsilon']:.2f}, {len(clients)} clients, "
          f"CV accuracy {evaluated['accuracy']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
