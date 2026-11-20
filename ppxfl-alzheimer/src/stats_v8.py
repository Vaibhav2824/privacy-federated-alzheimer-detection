"""
stats_v8.py -- paired significance tests: private (eps = 1) vs non-private.

Reads the out-of-fold prediction files written by
``experiments_v3 --save-oof DIR privacy`` (one .npz per label set, budget and
seed, subject IDs included) and compares the two models on the SAME subjects:

* McNemar test on correct / incorrect (chi-square with continuity correction,
  the same statistic as statsmodels ``mcnemar(exact=False, correction=True)``),
  with the discordant counts b and c;
* CN vs AD: DeLong test for two correlated ROC curves (fast DeLong, Sun & Xu
  2014), checked against a paired bootstrap on seed 42;
* CN/MCI/AD: paired subject-level bootstrap of the macro one-vs-rest AUROC
  difference;
* paired bootstrap 95% CI of the accuracy difference, and Wilson 95% CIs;
* accuracy by sex for the eps = 1 model.

Only aggregates are written (stats_summary.json, STATS_RESULTS.md); the .npz
inputs carry subject IDs and stay local (ADNI data use agreement).

    python -m src.stats_v8 --oof results_v8/paper_fixes/oof_LOCAL_ONLY \
        --features data/v8/features_v8_dp27.npz --out results_v8/paper_fixes
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

N_BOOT = 2000


# ---- tests ----------------------------------------------------------------------

def mcnemar(correct_a, correct_b) -> dict:
    """Continuity-corrected McNemar test. b: only A correct, c: only B correct."""
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b, c = int(np.sum(a & ~b_)), int(np.sum(~a & b_))
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(statistic),
            "p": float(stats.chi2.sf(statistic, 1))}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion k / n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(centre - half), float(centre + half))


def _midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y_true, score_a, score_b) -> dict:
    """Fast DeLong test (Sun & Xu 2014) for two correlated AUROCs on the same subjects."""
    y = np.asarray(y_true).astype(bool)
    scores = np.vstack([score_a, score_b]).astype(float)
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(s) for s in pos])
    ty = np.array([_midrank(s) for s in neg])
    tz = np.array([_midrank(s) for s in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z)))}


def macro_auroc(y_true, scores) -> float:
    if scores.shape[1] == 2:
        return float(roc_auc_score(y_true, scores[:, 1]))
    return float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))


def paired_bootstrap(y_true, scores_a, scores_b, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Paired subject-level bootstrap of accuracy and macro-AUROC differences (A - B)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    pred_a, pred_b = scores_a.argmax(1), scores_b.argmax(1)
    acc_diff, auc_diff = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < scores_a.shape[1]:
            continue
        acc_diff.append(np.mean(pred_a[idx] == y[idx]) - np.mean(pred_b[idx] == y[idx]))
        auc_diff.append(macro_auroc(y[idx], scores_a[idx]) - macro_auroc(y[idx], scores_b[idx]))
    acc_diff, auc_diff = np.asarray(acc_diff), np.asarray(auc_diff)

    def two_sided(d):
        return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))

    return {"acc_diff_ci": [float(np.percentile(acc_diff, 2.5)),
                            float(np.percentile(acc_diff, 97.5))],
            "auc_diff_ci": [float(np.percentile(auc_diff, 2.5)),
                            float(np.percentile(auc_diff, 97.5))],
            "auc_p": two_sided(auc_diff), "n_resamples": int(len(acc_diff))}


def compare(y, scores_a, scores_b, seed_check: bool = False) -> dict:
    """Model A (eps = 1) against model B (reference) on the same subjects."""
    pred_a, pred_b = scores_a.argmax(1), scores_b.argmax(1)
    out = {"n": int(len(y)),
           "acc_a": float(np.mean(pred_a == y)), "acc_b": float(np.mean(pred_b == y)),
           "auroc_a": macro_auroc(y, scores_a), "auroc_b": macro_auroc(y, scores_b),
           "mcnemar": mcnemar(pred_b == y, pred_a == y),
           "bootstrap": paired_bootstrap(y, scores_a, scores_b)}
    out["acc_diff"] = out["acc_a"] - out["acc_b"]
    out["wilson_a"] = wilson(int(np.sum(pred_a == y)), len(y))
    out["wilson_b"] = wilson(int(np.sum(pred_b == y)), len(y))
    if scores_a.shape[1] == 2:
        out["auroc_test"] = {"method": "DeLong", **delong(y, scores_a[:, 1], scores_b[:, 1])}
        if seed_check:
            out["delong_vs_bootstrap_p"] = out["bootstrap"]["auc_p"]
    else:
        out["auroc_test"] = {"method": "paired bootstrap", "p": out["bootstrap"]["auc_p"]}
    return out


# ---- driver (reads subject-level files; exercised on the local data only) ------------

def _load(path):  # pragma: no cover
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def run(oof_dir: str, features: str, out_dir: str) -> dict:  # pragma: no cover
    feats = np.load(features, allow_pickle=True)
    names = [str(n) for n in feats["feature_names"]]
    sex_of = dict(zip((str(s) for s in feats["subjects"]), feats["X"][:, names.index("demo::sex")]))
    result = {"tables": {"A": [], "B": []}, "gate": {}}
    for label_set in ("cn-ad", "cn-mci-ad"):
        by_sex = {"female": [], "male": []}
        for seed in (42, 123, 2024):
            private = _load(os.path.join(oof_dir, f"dp_{label_set}_eps1.0_s{seed}.npz"))
            plain = _load(os.path.join(oof_dir, f"dp_{label_set}_nonprivate_s{seed}.npz"))
            assert (private["subjects"] == plain["subjects"]).all()
            assert (private["y_true"] == plain["y_true"]).all()
            row = compare(private["y_true"], private["scores"], plain["scores"],
                          seed_check=(seed == 42))
            row.update(label_set=label_set, seed=seed)
            result["tables"]["A"].append(row)
            correct = private["predictions"] == private["y_true"]
            sex = np.array([sex_of.get(str(s), np.nan) for s in private["subjects"]])
            for name, code in (("male", 1.0), ("female", 2.0)):
                m = sex == code
                by_sex[name].append((int(m.sum()), int(correct[m].sum())))
        for name, rows in by_sex.items():
            n = int(round(np.mean([r[0] for r in rows])))
            acc = float(np.mean([k / max(nn, 1) for nn, k in rows]))
            result["tables"]["B"].append({
                "label_set": label_set, "sex": name, "n": n, "accuracy": acc,
                "wilson": wilson(int(round(acc * n)), n),
                "n_sex_unknown": int(np.sum(np.isnan([sex_of.get(str(s), np.nan)
                                                      for s in private["subjects"]])))})
    for label_set in ("cn-ad", "cn-mci-ad"):
        rows = [r for r in result["tables"]["A"] if r["label_set"] == label_set]
        result["gate"][label_set] = {"nonprivate": float(np.mean([r["acc_b"] for r in rows])),
                                     "eps1": float(np.mean([r["acc_a"] for r in rows]))}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stats_summary.json"), "w", encoding="utf-8") as h:
        json.dump(result, h, indent=2)
    return result


def markdown(result: dict, paper: dict) -> str:  # pragma: no cover
    task = {"cn-ad": "CN vs AD", "cn-mci-ad": "CN/MCI/AD"}
    lines = ["# Fix 1: private (eps = 1) vs non-private, same subjects", "",
             "## Reproduction gate", "",
             "| Task | Non-private (paper) | Non-private (rerun) | eps=1 (paper) | eps=1 (rerun) |",
             "|---|---|---|---|---|"]
    ok = True
    for ls, g in result["gate"].items():
        p = paper[ls]
        ok &= abs(100 * g["nonprivate"] - p[0]) <= 0.3 and abs(100 * g["eps1"] - p[1]) <= 0.3
        lines.append(f"| {task[ls]} | {p[0]:.1f} | {100 * g['nonprivate']:.1f} | {p[1]:.1f} | "
                     f"{100 * g['eps1']:.1f} |")
    lines += ["", f"Gate passed (within ±0.3 points): **{'yes' if ok else 'NO'}**", "",
              "## Table A", "",
              "| Task | Seed | Acc non-private | Acc ε=1 | Acc diff [95% CI] | McNemar b / c | "
              "McNemar p | AUROC non-private | AUROC ε=1 | AUROC test p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["tables"]["A"]:
        ci = r["bootstrap"]["acc_diff_ci"]
        test = r["auroc_test"]
        extra = (f" (bootstrap {r['delong_vs_bootstrap_p']:.3f})"
                 if "delong_vs_bootstrap_p" in r else "")
        lines.append(
            f"| {task[r['label_set']]} | {r['seed']} | {100 * r['acc_b']:.1f} | {100 * r['acc_a']:.1f} | "
            f"{100 * r['acc_diff']:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] | "
            f"{r['mcnemar']['b']} / {r['mcnemar']['c']} | {r['mcnemar']['p']:.3f} | "
            f"{r['auroc_b']:.3f} | {r['auroc_a']:.3f} | {test['method']}: {test['p']:.3f}{extra} |")
    lines += ["", "b: subjects only the non-private model got right; c: only the ε=1 model.", "",
              "## Table B: accuracy by sex, ε = 1 model (mean over seeds)", "",
              "| Task | Sex | n | Accuracy | Wilson 95% CI |", "|---|---|---|---|---|"]
    for r in result["tables"]["B"]:
        lo, hi = r["wilson"]
        lines.append(f"| {task[r['label_set']]} | {r['sex']} | {r['n']} | {100 * r['accuracy']:.1f} | "
                     f"[{100 * lo:.1f}, {100 * hi:.1f}] |")
    unknown = {r["label_set"]: r["n_sex_unknown"] for r in result["tables"]["B"]}
    lines += ["", "Subjects with sex not recorded (excluded from Table B): "
              + ", ".join(f"{task[k]} {v}" for k, v in unknown.items()) + "."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:  # pragma: no cover
    p = argparse.ArgumentParser(description="paired tests, eps=1 vs non-private")
    p.add_argument("--oof", default=os.path.join("results_v8", "paper_fixes", "oof_LOCAL_ONLY"))
    p.add_argument("--features", default=os.path.join("data", "v8", "features_v8_dp27.npz"))
    p.add_argument("--out", default=os.path.join("results_v8", "paper_fixes"))
    a = p.parse_args(argv)
    if not glob.glob(os.path.join(a.oof, "*.npz")):
        raise SystemExit(f"no out-of-fold files in {a.oof}")
    result = run(a.oof, a.features, a.out)
    paper = {"cn-ad": (88.5, 89.5), "cn-mci-ad": (61.4, 61.8)}
    with open(os.path.join(a.out, "STATS_RESULTS.md"), "w", encoding="utf-8") as h:
        h.write(markdown(result, paper))
    print(open(os.path.join(a.out, "STATS_RESULTS.md"), encoding="utf-8").read())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
