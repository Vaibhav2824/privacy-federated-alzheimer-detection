"""
phase_shift_v8.py -- does a model trained on older ADNI phases work on ADNI4?

ADNI4 (2023 onward) uses newer scanners and protocols that no v7 subject came
from.  Two numbers answer the question, on the same ADNI4 subjects:

* shift:  train on ADNI1/GO/2/3 only, test on every ADNI4 subject;
* mixed:  5-fold CV over the whole cohort, scored on the ADNI4 subjects only,
          so ADNI4 subjects from other folds are in each training set.

If "mixed" beats "shift", seeing newer scanners in training is what fixed it.
The same model and preprocessing as evaluate_v3 are used throughout.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from .build_v8 import phase_of
from .evaluate_v3 import LABEL_SETS, build_model, load_features


def run(features: str, model_name: str, label_set: str, seeds) -> dict:
    X, y, subjects, _sites, _names = load_features(features)
    classes = LABEL_SETS[label_set]["classes"]
    keep = np.isin(y, classes)
    X, y = X[keep], y[keep]
    subjects = [s for s, k in zip(subjects, keep) if k]
    new = np.array([phase_of(s) == "ADNI4" for s in subjects])
    out = {"n_adni4": int(new.sum()), "n_older": int((~new).sum()), "shift": [], "mixed": []}

    for seed in seeds:
        model = build_model(model_name, seed).fit(X[~new], y[~new])
        pred = model.predict(X[new])
        out["shift"].append({"accuracy": accuracy_score(y[new], pred),
                             "balanced_accuracy": balanced_accuracy_score(y[new], pred)})

        oof = np.full(len(y), -1)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            oof[te] = build_model(model_name, seed).fit(X[tr], y[tr]).predict(X[te])
        out["mixed"].append({"accuracy": accuracy_score(y[new], oof[new]),
                             "balanced_accuracy": balanced_accuracy_score(y[new], oof[new])})

    for k in ("shift", "mixed"):
        out[f"{k}_mean"] = {m: float(np.mean([r[m] for r in out[k]]))
                            for m in ("accuracy", "balanced_accuracy")}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ADNI4 scanner-shift test")
    p.add_argument("--features", default="data/v8/features_v8_mri.npz")
    p.add_argument("--model", default="logreg_cv")
    p.add_argument("--label-sets", nargs="+", default=["cn-ad", "cn-mci-ad"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    p.add_argument("--out", default="results_v8/phase_shift.json")
    a = p.parse_args(argv)
    res = {ls: run(a.features, a.model, ls, a.seeds) for ls in a.label_sets}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as h:
        json.dump(res, h, indent=2)
    for ls, r in res.items():
        print(f"{ls}: ADNI4 n={r['n_adni4']}  shift acc={r['shift_mean']['accuracy']:.3f}  "
              f"mixed acc={r['mixed_mean']['accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
