"""
export_site.py — every number the multi-page site shows, in one file.

The site renders ``ui/public/data/site.json`` and nothing else, so no figure is
typed into a page by hand.  Only aggregates leave this script: per-condition
means, per-client counts and the historical headline of each earlier cohort.
No subject identifier, site identifier or per-subject value is written.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from statistics import mean

METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc",
           "explanation_agreement", "medial_temporal_share")


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _condition(c):
    out = {k: c[k] for k in ("family", "model", "label_set", "split_scheme", "client_scheme",
                             "target_epsilon", "n_runs", "n_subjects", "chance_accuracy",
                             "noise_multiplier", "perturbed_dimension") if k in c}
    for key in METRICS:
        if isinstance(c.get(key), dict) and c[key].get("mean") is not None:
            out[key] = c[key]["mean"]
            out[f"{key}_std"] = c[key].get("std")
    return out


def _headline(summary_path, label_set="cn-mci-ad", split="subject", model="ensemble"):
    """Three-class subject-disjoint accuracy of one earlier cohort."""
    for c in _load(summary_path)["conditions"]:
        if (c["family"], c["model"], c["label_set"], c["split_scheme"]) == \
                ("centralised", model, label_set, split):
            return {"accuracy": c["accuracy"]["mean"], "n_subjects": c["n_subjects"]}
    return None


def _agreement_eps10(summary_path):
    """CN-vs-AD attribution agreement at eps 10 for one cohort's summary."""
    return next(
        c["explanation_agreement"]["mean"] for c in _load(summary_path)["conditions"]
        if c["family"] == "privacy" and c["label_set"] == "cn-ad" and c["target_epsilon"] == 10.0)


def _history(root):
    """v1 slice-leak headline, v2 subject-disjoint rebuild, then v3 to v6."""
    with open(os.path.join(root, "results", "metrics", "all_experiments_results.csv"),
              encoding="utf-8") as handle:
        v1 = max(float(r["accuracy"]) for r in csv.DictReader(handle)
                 if r["experiment"].startswith("resnet50_fedavg"))
    v2 = mean(_load(p)["accuracy"] for p in
              glob.glob(os.path.join(root, "results_v2", "metrics", "resnet50_centralised_*_metrics.json")))
    v3_path = os.path.join(root, "results_v3", "results_summary_v3.json")
    v5_path = os.path.join(root, "results_v5", "results_summary_v5.json")
    # The agreement at eps 10 on the first cohort and on the same pipeline at
    # 799 subjects: the finding the integrity page reports as downgraded.
    v3 = {**_headline(v3_path), "agreement_eps10": _agreement_eps10(v3_path)}
    v5 = {**_headline(v5_path), "agreement_eps10": _agreement_eps10(v5_path)}
    return {
        "v1": {"accuracy": v1},
        "v2": {"accuracy": v2},
        "v3": v3,
        "v4": _headline(os.path.join(root, "results_v4", "results_summary_v4.json")),
        "v5": v5,
        "v6": _headline(os.path.join(root, "results_v6", "results_summary_v6_imaging_only.json"),
                        model="ensemble_cv"),
    }


def build(root: str) -> dict:
    results = os.path.join(root, "results_v6")
    metrics = os.path.join(results, "metrics")
    run = _load(os.path.join(metrics, "dp_cn-mci-ad_eps1.0_s42.json"))
    partition = run["partition"]
    clients = sorted(
        ({"n_subjects": c["n_subjects"], "n_sites": len(c["sites"]),
          "class_counts": [c["class_counts"].get(str(k), 0) for k in range(3)],
          "label_entropy": c["label_entropy"]} for c in partition["clients"]),
        key=lambda c: -c["n_subjects"],
    )
    attacks = {}
    for r in _load(os.path.join(metrics, "mia_v3.json"))["runs"]:
        key = "none" if r["target_epsilon"] is None else str(float(r["target_epsilon"]))
        attacks.setdefault(f"{r['label_set']}|{key}", []).append(r["attack_auroc"])
    # Centralised headline numbers are imaging only, like the federated and
    # private arms; the age-and-sex variant is kept as its own family.
    imaging = _load(os.path.join(results, "results_summary_v6_imaging_only.json"))["conditions"]
    with_covariates = _load(os.path.join(results, "results_summary_v6.json"))["conditions"]
    conditions = [_condition(c) for c in imaging if c["family"] == "centralised"]
    for c in with_covariates:
        row = _condition(c)
        if row["family"] == "centralised":
            row["family"] = "centralised+cov"
        conditions.append(row)
    return {
        "cohort": "v6",
        "conditions": conditions,
        "attack_auroc": {k: mean(v) for k, v in attacks.items()},
        "confounds": _load(os.path.join(metrics, "confounds.json")),
        "dimension_law": _load(os.path.join(metrics, "dimension_law.json"))["rows"],
        "dp_config": {k: run[k] for k in ("clip_norm", "rounds", "local_epochs",
                                          "subject_sample_rate", "delta")},
        "partition": {"n_clients": partition["n_clients"],
                      "n_sites": sum(c["n_sites"] for c in clients),
                      "clients": clients},
        "history": _history(root),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="export the site's data file")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=os.path.join("..", "ui", "public", "data", "site.json"))
    args = parser.parse_args(argv)
    data = build(args.root)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
    print(f"wrote {args.out}: {len(data['conditions'])} conditions, "
          f"{data['partition']['n_clients']} clients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
