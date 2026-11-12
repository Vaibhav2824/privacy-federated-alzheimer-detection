"""
collect_v8_summary.py -- the shareable part of results_v8/.

results_v8/ stays local: its per-fold files carry ADNI site codes and the logs
are scratch.  This copies only cohort-level aggregates (means over folds and
seeds, attack AUROCs, the dimension law, the ADNI4 shift test) into
results_v8_summary/ and writes one headline file, results_summary_v8.json,
that sets the v8 private model beside the paper's v7 model.

    python scripts/collect_v8_summary.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
from statistics import mean, stdev

SRC, OUT = "results_v8", "results_v8_summary"
COPY = {
    "summary_metrics_dp27.json": "privacy_dp27.json",
    "summary_metrics_mri.json": "privacy_paper_model_on_v8.json",
    "metrics_dp27/mia_v3.json": "membership_inference_dp27.json",
    "metrics_dp27/dimension_law.json": "dimension_law_dp27.json",
    "metrics_dp27_fedavg/federated_summary.json": "fedavg_dp27.json",
    "metrics_dp27_fedprox/federated_summary.json": "fedprox_dp27.json",
    "metrics_fedboost/fedboost_summary.json": "fedboost.json",
    "metrics_central_mri/centralised_summary.json": "centralised_mri.json",
    "metrics_central_bio/centralised_summary.json": "centralised_bio.json",
    "phase_shift_mri.json": "adni4_shift_mri.json",
    "phase_shift_bio.json": "adni4_shift_bio.json",
}
SUBJECT_ID = re.compile(r"\d{3}_S_\d{4}")


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _privacy(summary, label_set, eps):
    c = next(c for c in summary["conditions"] if c["family"] == "privacy"
             and c["label_set"] == label_set and c["target_epsilon"] == eps)
    out = {k: round(c[k]["mean"], 4) for k in ("accuracy", "balanced_accuracy", "macro_auroc")}
    out["accuracy_std"] = round(c["accuracy"]["std"], 4)
    if "explanation_agreement" in c:
        out["explanation_agreement"] = round(c["explanation_agreement"]["mean"], 3)
    return out


def _runs(rows, prefix, key=None):
    acc = [(r[key] if key else r)["accuracy"] for r in rows if r["tag"].startswith(prefix)]
    return {"accuracy": round(mean(acc), 4), "accuracy_std": round(stdev(acc), 4), "n_runs": len(acc)}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for src, dst in COPY.items():
        text = open(os.path.join(SRC, src), encoding="utf-8").read()
        if SUBJECT_ID.search(text):
            raise SystemExit(f"{src} contains a subject identifier; not copied")
        shutil.copyfile(os.path.join(SRC, src), os.path.join(OUT, dst))

    v7 = _load(os.path.join("results_v7", "results_summary_v7.json"))
    dp27 = _load(os.path.join(SRC, "summary_metrics_dp27.json"))
    mri = _load(os.path.join(SRC, "summary_metrics_mri.json"))
    mia = _load(os.path.join(SRC, "metrics_dp27", "mia_v3.json"))["runs"]
    fedavg = _load(os.path.join(SRC, "metrics_dp27_fedavg", "federated_summary.json"))["runs"]
    fedprox = _load(os.path.join(SRC, "metrics_dp27_fedprox", "federated_summary.json"))["runs"]
    boost = _load(os.path.join(SRC, "metrics_fedboost", "fedboost_summary.json"))

    def mia_auroc(eps):
        vals = [r["attack_auroc"] for r in mia if r["label_set"] == "cn-mci-ad"
                and r["target_epsilon"] == eps]
        return round(mean(vals), 3)

    headline = {
        "protocol": "5-fold subject-disjoint CV x seeds 42/123/2024, real ADNI sites as "
                    "clients, subject-level DP-FedAvg, PRV accountant, delta = 1e-3",
        "main_model": {"name": "v8 DP-FedAvg logistic regression, 27 a-priori features",
                       "parameters_three_class": 84},
        "private": {},
        "membership_inference_auroc_three_class": {
            "no_dp": mia_auroc(None), "eps_10": mia_auroc(10.0), "eps_1": mia_auroc(1.0)},
        "federated_no_dp": {
            "fedavg_dp27": {ls: _runs(fedavg, f"fed_natural_{ls}_subject") for ls in ("cn-mci-ad", "cn-ad")},
            "fedprox_dp27": {ls: _runs(fedprox, f"fed_natural_{ls}_subject") for ls in ("cn-mci-ad", "cn-ad")},
            "fedboost": {ls: _runs(boost, f"fedboost_{ls}_subject", "fedboost") for ls in ("cn-mci-ad", "cn-ad")},
        },
    }
    for ls in ("cn-mci-ad", "cn-ad"):
        for eps in (1.0, 10.0):
            headline["private"][f"{ls}_eps{eps:g}"] = {
                "v7_paper_model": _privacy(v7, ls, eps),
                "paper_model_on_v8_data": _privacy(mri, ls, eps),
                "v8_dp27": _privacy(dp27, ls, eps),
            }
    with open(os.path.join(OUT, "results_summary_v8.json"), "w", encoding="utf-8") as handle:
        json.dump(headline, handle, indent=2)
    print(f"wrote {OUT}/ ({len(COPY) + 1} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
