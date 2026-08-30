"""
confounds_v3.py — what a model could be reading instead of anatomy.

An imaging accuracy is only evidence about imaging if the cohort's demographics
could not have produced it on their own.  In the v3 cohort they partly could:
CN is 66% female while MCI is 62% male, so a rule that guesses from sex alone
already beats the chance rate.  Any headline accuracy measured on that sample
is, to an unknown degree, a sex classifier.

This module measures the degree rather than arguing about it.  It fits
demographics-only baselines under exactly the protocol used for the imaging
models -- same folds, same seeds, same metrics -- so the numbers are
comparable, and then asks the question that actually matters: does the imaging
representation add anything *beyond* demographics, and does it still work
inside a single sex stratum where sex cannot help at all?

Three baselines are reported:

``sex``, ``age``, ``sex+age``
    What the demographics alone buy.  If the imaging model does not clearly
    exceed these, it has not been shown to use anatomy.

``imaging`` and ``imaging+demographics``
    The increment from anatomy, and whether demographics still add anything
    once anatomy is available.

``stratified``
    Imaging accuracy computed within the female and male strata separately.
    Sex is constant inside a stratum, so it cannot contribute, and a result
    that survives here is not a demographic artefact.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os

import numpy as np

from .evaluate_v3 import LABEL_SETS, load_features
from .splits_v3 import make_folds


def load_demographics(path: str) -> dict:
    """Subject -> sex and age, from an IDA advanced-search export."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            subject = row.get("Subject ID")
            if not subject or subject in out:
                continue
            try:
                age = float(row["Age"])
            except (TypeError, ValueError, KeyError):
                age = None
            out[subject] = {"sex": row.get("Sex"), "age": age}
    return out


def _metrics(y_true, y_pred, y_score, n_classes):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 roc_auc_score)

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


def _fit_predict(train_X, train_y, test_X, seed: int, n_classes: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced",
                                   random_state=seed)),
    ])
    model.fit(train_X, train_y)
    score = model.predict_proba(test_X)
    return np.argmax(score, axis=1), score


def evaluate_matrix(X, y, sites, label_set: str, scheme: str, n_splits: int,
                    seeds) -> dict:
    """Cross-validated evaluation of one design matrix."""
    classes = LABEL_SETS[label_set]["classes"]
    n_classes = len(classes)
    per_seed = []
    for seed in seeds:
        folds = make_folds(scheme, y, sites, n_splits=n_splits, seed=seed)
        pred = np.zeros(len(y), dtype=int)
        score = np.zeros((len(y), n_classes), dtype=float)
        for fold in folds:
            p, s = _fit_predict(X[fold.train], y[fold.train], X[fold.test],
                                seed, n_classes)
            pred[fold.test] = p
            score[fold.test] = s
        per_seed.append(_metrics(y, pred, score, n_classes))
    return {
        key: {
            "mean": float(np.mean([m[key] for m in per_seed])),
            "std": float(np.std([m[key] for m in per_seed])),
        }
        for key in per_seed[0]
    }


def build_designs(features, sex, age):
    """The design matrices whose comparison answers the confound question."""
    sex_col = np.asarray([[1.0 if s == "F" else 0.0] for s in sex])
    age_col = np.asarray([[a] for a in age])
    return {
        "sex": sex_col,
        "age": age_col,
        "sex+age": np.hstack([sex_col, age_col]),
        "imaging": features,
        "imaging+demographics": np.hstack([features, sex_col, age_col]),
    }


def run(features_path: str, demographics_path: str, label_sets, scheme: str,
        n_splits: int, seeds) -> dict:
    X, y, subjects, sites, _ = load_features(features_path)
    demographics = load_demographics(demographics_path)

    keep = [i for i, s in enumerate(subjects)
            if s in demographics and demographics[s]["age"] is not None
            and demographics[s]["sex"] in ("M", "F")]
    if len(keep) < len(subjects):
        print(f"demographics available for {len(keep)}/{len(subjects)} subjects; "
              "the rest are excluded from this analysis")

    X = X[keep]
    y = y[keep]
    subjects = [subjects[i] for i in keep]
    sites = [sites[i] for i in keep]
    sex = [demographics[s]["sex"] for s in subjects]
    age = [demographics[s]["age"] for s in subjects]

    report = {
        "n_subjects": len(subjects),
        "cohort_composition": {},
        "results": {},
        "stratified": {},
    }

    counts = collections.Counter(zip((int(v) for v in y), sex))
    names = {0: "CN", 1: "MCI", 2: "AD"}
    report["cohort_composition"] = {
        f"{names[k]}-{s}": n for (k, s), n in sorted(counts.items())
    }

    for label_set in label_sets:
        classes = LABEL_SETS[label_set]["classes"]
        mask = np.isin(y, classes)
        remap = {c: i for i, c in enumerate(classes)}
        y_sub = np.asarray([remap[v] for v in y[mask]])
        sites_sub = [s for s, m in zip(sites, mask) if m]
        sex_sub = [s for s, m in zip(sex, mask) if m]
        age_sub = [a for a, m in zip(age, mask) if m]

        designs = build_designs(X[mask], sex_sub, age_sub)
        report["results"][label_set] = {}
        for name, matrix in designs.items():
            report["results"][label_set][name] = evaluate_matrix(
                matrix, y_sub, sites_sub, label_set, scheme, n_splits, seeds
            )
            value = report["results"][label_set][name]["balanced_accuracy"]["mean"]
            print(f"  {label_set:<10} {name:<22} balanced acc = {value * 100:5.1f}%",
                  flush=True)

        # Within a single sex stratum, sex is constant and cannot contribute.
        report["stratified"][label_set] = {}
        for stratum in ("F", "M"):
            index = [i for i, s in enumerate(sex_sub) if s == stratum]
            if len(index) < 40:
                continue
            report["stratified"][label_set][stratum] = evaluate_matrix(
                X[mask][index], y_sub[index], [sites_sub[i] for i in index],
                label_set, scheme, n_splits, seeds
            )
            report["stratified"][label_set][stratum]["n"] = len(index)
            value = report["stratified"][label_set][stratum]["balanced_accuracy"]["mean"]
            print(f"  {label_set:<10} imaging, {stratum} only (n={len(index)})"
                  f"      balanced acc = {value * 100:5.1f}%", flush=True)

    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="demographic confound analysis")
    parser.add_argument("--features", default=os.path.join("data", "mni2mm", "features.npz"))
    parser.add_argument("--demographics", default=os.path.join("data", "ida_search_v4.csv"))
    parser.add_argument("--out", default=os.path.join("results_v3", "metrics",
                                                      "confounds.json"))
    parser.add_argument("--label-sets", nargs="+", default=["cn-mci-ad", "cn-ad"])
    parser.add_argument("--scheme", default="subject")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    args = parser.parse_args(argv)

    report = run(args.features, args.demographics, args.label_sets, args.scheme,
                 args.n_splits, args.seeds)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
