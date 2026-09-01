"""
evaluate_v3.py — centralised baselines on the MNI-registered cohort.

Runs the atlas-morphometry representation through a small set of classifiers
under both split schemes and both label sets, and writes one JSON per
configuration.  The point of this module is to establish what the cohort
actually supports before any federated or private variant is layered on top:
a federated result is only interpretable relative to a centralised number
measured the same way.

Every configuration reports accuracy, macro F1 and macro AUROC with a
bootstrap interval, together with the chance rate for its label set, so that a
result near chance is visible as such rather than quoted as an accuracy.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .splits_v3 import make_folds

LABEL_SETS = {
    "cn-mci-ad": {"classes": (0, 1, 2), "names": ("CN", "MCI", "AD")},
    "cn-ad": {"classes": (0, 2), "names": ("CN", "AD")},
    "cn-mci": {"classes": (0, 1), "names": ("CN", "MCI")},
    "mci-ad": {"classes": (1, 2), "names": ("MCI", "AD")},
}


def restrict_to_cohort(X, y, subjects, sites, cohort: str, features_path: str,
                       demographics_path: str, seed: int = 42):
    """Subset to the named analysis cohort, or pass everything through.

    ``balanced`` removes the sex confound by construction rather than adjusting
    for it afterwards; ``full`` keeps every QC-passing subject.  Both are
    reported, since they answer different questions -- see analysis_cohort.
    """
    if cohort == "all":
        return X, y, subjects, sites
    import os

    from .analysis_cohort import balanced_indices, load_demographics

    if not os.path.exists(demographics_path):
        return X, y, subjects, sites
    demographics = load_demographics(demographics_path)
    if cohort == "full":
        index = [i for i, s in enumerate(subjects) if s in demographics]
    elif cohort == "balanced":
        index = balanced_indices(subjects, y, demographics, seed=seed)
    else:
        raise ValueError(f"unknown cohort: {cohort}")
    if not index:
        # Falling back silently would report a full-cohort number under a
        # balanced-cohort heading, which is the one confusion this module
        # exists to prevent.
        print(f"[cohort] {cohort!r} selected no subjects; "
              "reporting on the whole cohort instead")
        return X, y, subjects, sites
    return (X[index], y[index], [subjects[i] for i in index],
            [sites[i] for i in index])


def load_features(path: str):
    data = np.load(path, allow_pickle=True)
    return (
        data["X"].astype(np.float64),
        data["y"].astype(int),
        list(data["subjects"]),
        list(data["sites"]),
        list(data["feature_names"]),
    )


def build_model(name: str, seed: int):
    """Classifiers sized for a few hundred subjects and ~140 features."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1, max_iter=5000, class_weight="balanced", random_state=seed
            )),
        ])
    if name == "svm":
        from sklearn.calibration import CalibratedClassifierCV

        return Pipeline([
            ("scale", StandardScaler()),
            # ``SVC(probability=True)`` is deprecated from scikit-learn 1.9;
            # calibrating explicitly is the documented replacement and gives the
            # same thing — probabilities that can go into a macro AUROC.
            ("clf", CalibratedClassifierCV(
                SVC(C=1.0, kernel="rbf", gamma="scale",
                    class_weight="balanced", random_state=seed),
                ensemble=False, cv=3,
            )),
        ])
    if name == "lda":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced",
            random_state=seed, n_jobs=-1
        )
    if name == "ensemble":
        from sklearn.ensemble import VotingClassifier

        # Soft voting over the four, which are decorrelated enough to help: a
        # shrinkage-regularised linear boundary, an RBF boundary, and an axis
        # aligned tree ensemble make different errors on region features.
        return VotingClassifier(
            estimators=[(n, build_model(n, seed)) for n in ("logreg", "svm", "lda", "rf")],
            voting="soft",
        )
    raise ValueError(f"unknown model: {name}")


def _metrics(y_true, y_pred, y_score, n_classes: int) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
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


def _bootstrap(y_true, y_pred, y_score, n_classes: int, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    keys = ("accuracy", "macro_f1", "balanced_accuracy", "macro_auroc")
    samples = {k: [] for k in keys}
    size = len(y_true)
    for _ in range(n):
        index = rng.integers(0, size, size)
        if len(np.unique(y_true[index])) < 2:
            continue
        values = _metrics(y_true[index], y_pred[index], y_score[index], n_classes)
        for key in keys:
            samples[key].append(values[key])
    return {
        f"{key}_ci95": [
            float(np.nanpercentile(samples[key], 2.5)),
            float(np.nanpercentile(samples[key], 97.5)),
        ]
        for key in keys if samples[key]
    }


def run_configuration(X, y, sites, model_name: str, label_set: str, scheme: str,
                      n_splits: int, seed: int) -> dict:
    """Cross-validated out-of-fold evaluation for one configuration."""
    classes = LABEL_SETS[label_set]["classes"]
    keep = np.isin(y, classes)
    X_sub, y_sub = X[keep], y[keep]
    sites_sub = [s for s, k in zip(sites, keep) if k]
    remap = {c: i for i, c in enumerate(classes)}
    y_sub = np.asarray([remap[v] for v in y_sub])

    folds = make_folds(scheme, y_sub, sites_sub, n_splits=n_splits, seed=seed)
    n_classes = len(classes)
    oof_pred = np.zeros(len(y_sub), dtype=int)
    oof_score = np.zeros((len(y_sub), n_classes), dtype=float)
    per_fold = []

    for fold in folds:
        model = build_model(model_name, seed)
        model.fit(X_sub[fold.train], y_sub[fold.train])
        score = model.predict_proba(X_sub[fold.test])
        # Labels were remapped to 0..n-1 before fitting, so ``classes_`` is that
        # range in order and the argmax column index is the predicted label.
        pred = np.argmax(score, axis=1)
        oof_pred[fold.test] = pred
        oof_score[fold.test] = score
        per_fold.append(_metrics(y_sub[fold.test], pred, score, n_classes))

    overall = _metrics(y_sub, oof_pred, oof_score, n_classes)
    overall.update(_bootstrap(y_sub, oof_pred, oof_score, n_classes, seed=seed))

    counts = np.bincount(y_sub, minlength=n_classes)
    return {
        "model": model_name,
        "label_set": label_set,
        "class_names": list(LABEL_SETS[label_set]["names"]),
        "split_scheme": scheme,
        "n_splits": n_splits,
        "seed": seed,
        "n_subjects": int(len(y_sub)),
        "class_counts": {
            name: int(c) for name, c in zip(LABEL_SETS[label_set]["names"], counts)
        },
        "chance_accuracy": float(counts.max() / counts.sum()),
        "uniform_chance": 1.0 / n_classes,
        "overall": overall,
        "per_fold": per_fold,
        "fold_mean": {
            key: float(np.mean([f[key] for f in per_fold]))
            for key in per_fold[0]
        },
        "fold_std": {
            key: float(np.std([f[key] for f in per_fold]))
            for key in per_fold[0]
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="centralised baselines, v3 cohort")
    parser.add_argument("--features", default=os.path.join("data", "mni2mm", "features.npz"))
    parser.add_argument("--out-dir", default=os.path.join("results_v3", "metrics"))
    parser.add_argument("--models", nargs="+", default=["logreg", "svm", "lda", "rf", "ensemble"])
    parser.add_argument("--label-sets", nargs="+", default=list(LABEL_SETS))
    parser.add_argument("--schemes", nargs="+", default=["subject", "site"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cohort", choices=["all", "full", "balanced"],
                        default="all",
                        help="balanced removes the sex confound by construction")
    parser.add_argument("--demographics",
                        default=os.path.join("data", "ida_search_v4.csv"))
    args = parser.parse_args(argv)

    X, y, subjects, sites, feature_names = load_features(args.features)
    X, y, subjects, sites = restrict_to_cohort(
        X, y, subjects, sites, args.cohort, args.features, args.demographics
    )
    print(f"{X.shape[0]} subjects, {X.shape[1]} features, "
          f"{len(set(sites))} sites")
    os.makedirs(args.out_dir, exist_ok=True)

    summary = []
    for label_set in args.label_sets:
        for scheme in args.schemes:
            for model_name in args.models:
                for seed in args.seeds:
                    result = run_configuration(
                        X, y, sites, model_name, label_set, scheme,
                        args.n_splits, seed
                    )
                    tag = f"{model_name}_{label_set}_{scheme}_s{seed}"
                    with open(os.path.join(args.out_dir, f"{tag}.json"), "w",
                              encoding="utf-8") as handle:
                        json.dump(result, handle, indent=2)
                    summary.append({
                        "tag": tag,
                        "accuracy": result["overall"]["accuracy"],
                        "balanced_accuracy": result["overall"]["balanced_accuracy"],
                        "macro_f1": result["overall"]["macro_f1"],
                        "macro_auroc": result["overall"]["macro_auroc"],
                        "chance": result["chance_accuracy"],
                    })
                    print(f"  {tag}: acc={result['overall']['accuracy']:.3f} "
                          f"bal={result['overall']['balanced_accuracy']:.3f} "
                          f"F1={result['overall']['macro_f1']:.3f} "
                          f"AUROC={result['overall']['macro_auroc']:.3f} "
                          f"(chance {result['chance_accuracy']:.3f})", flush=True)

    with open(os.path.join(args.out_dir, "centralised_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
