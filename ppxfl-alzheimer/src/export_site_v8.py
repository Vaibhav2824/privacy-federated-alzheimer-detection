"""
export_site_v8.py — the site's data file from the v8 results (1,262 subjects).

Same schema as export_site.py, with the v8 meaning of each family:

  centralised       134 MRI region volumes / TIV (imaging only)
  centralised+cov   the same regions plus the fluid and PET biomarkers
  federated         the 27-feature model over the real sites: FedAvg, FedProx, boosting
  privacy           the 27-feature, 84-parameter model under subject-level DP-FedAvg

It only reads finished results and writes aggregates: per-condition means,
per-client counts, sex-stratified accuracy and the ADNI-4 scanner shift. No
subject identifier, site identifier or per-subject value is written.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from statistics import mean, stdev

METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc")
TAG = re.compile(r"(?P<model>.+)_(?P<ls>cn-mci-ad|cn-ad|mci-ad|cn-mci)_(?P<split>subject|site)_s\d+$")


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sd(values):
    return stdev(values) if len(values) > 1 else 0.0


def _condition(c):
    """A summary condition (privacy family) flattened to means and stds."""
    out = {k: c[k] for k in ("family", "model", "label_set", "split_scheme", "client_scheme",
                             "target_epsilon", "n_runs", "n_subjects", "chance_accuracy",
                             "noise_multiplier", "perturbed_dimension") if k in c}
    for key in (*METRICS, "explanation_agreement", "medial_temporal_share"):
        if isinstance(c.get(key), dict) and c[key].get("mean") is not None:
            out[key] = c[key]["mean"]
            out[f"{key}_std"] = c[key].get("std")
    return out


def _runs(files, family, model=None, scheme="none", key="overall"):
    """Group per-seed run files by (model, task, split) into one condition each."""
    groups = {}
    for path in files:
        run = _load(path)
        tag = TAG.match(os.path.basename(path)[:-5])
        if not tag:
            continue
        name = model or run.get("model") or tag["model"]
        groups.setdefault((name, tag["ls"], tag["split"]), []).append(run)
    out = []
    for (name, ls, split), runs in sorted(groups.items()):
        row = {"family": family, "model": name, "label_set": ls, "split_scheme": split,
               "client_scheme": scheme, "target_epsilon": None, "n_runs": len(runs),
               "n_subjects": runs[0]["n_subjects"], "chance_accuracy": runs[0]["chance_accuracy"]}
        for metric in METRICS:
            values = [r[key][metric] for r in runs if metric in r.get(key, {})]
            if values:
                row[metric] = mean(values)
                row[f"{metric}_std"] = _sd(values)
        out.append(row)
    return out


def _fedboost(path, chance):
    """Federated boosting, stored as one list of per-seed results."""
    groups = {}
    for r in _load(path):
        tag = TAG.match(r["tag"])
        if tag:
            groups.setdefault((tag["ls"], tag["split"]), []).append(r["fedboost"])
    return [{"family": "federated", "model": "fedboost", "label_set": ls, "split_scheme": split,
             "client_scheme": "natural", "target_epsilon": None, "n_runs": len(runs),
             "n_subjects": chance[ls][0], "chance_accuracy": chance[ls][1],
             **{m: mean(r[m] for r in runs) for m in METRICS},
             **{f"{m}_std": _sd([r[m] for r in runs]) for m in METRICS}}
            for (ls, split), runs in sorted(groups.items())]


def build(root: str, summary: str, previous: dict) -> dict:
    v8 = os.path.join(root, "results_v8")
    centralised = _runs(glob.glob(os.path.join(v8, "metrics_central_mri", "*_s*.json")), "centralised")
    with_bio = _runs(glob.glob(os.path.join(v8, "metrics_central_bio", "*_s*.json")), "centralised+cov")
    chance = {c["label_set"]: (c["n_subjects"], c["chance_accuracy"]) for c in centralised}
    federated = (
        _runs(glob.glob(os.path.join(v8, "metrics_dp27_fedavg", "fed_*.json")), "federated", "fedavg", "natural")
        + _runs(glob.glob(os.path.join(v8, "metrics_dp27_fedprox", "fed_*.json")), "federated", "fedprox", "natural")
        + _fedboost(os.path.join(summary, "fedboost.json"), chance))
    privacy = [_condition(c) for c in _load(os.path.join(v8, "summary_metrics_dp27.json"))["conditions"]]

    attacks = {}
    for r in _load(os.path.join(v8, "metrics_dp27", "mia_v3.json"))["runs"]:
        key = "none" if r["target_epsilon"] is None else str(float(r["target_epsilon"]))
        attacks.setdefault(f"{r['label_set']}|{key}", []).append(r["attack_auroc"])

    run = _load(os.path.join(v8, "metrics_dp27", "dp_cn-mci-ad_eps1.0_s42.json"))
    fedavg = _load(os.path.join(v8, "metrics_dp27_fedavg", "fed_natural_cn-mci-ad_subject_s42.json"))
    clients = sorted(
        ({"n_subjects": c["n_subjects"], "n_sites": len(c["sites"]),
          "class_counts": [c["class_counts"].get(str(k), 0) for k in range(3)],
          "label_entropy": c["label_entropy"]} for c in run["partition"]["clients"]),
        key=lambda c: -c["n_subjects"])

    by_sex = {}
    for row in _load(os.path.join(v8, "paper_fixes", "stats_summary.json"))["tables"]["B"]:
        by_sex.setdefault(row["label_set"], {})[row["sex"]] = {
            "accuracy": row["accuracy"], "low": row["wilson"][0], "high": row["wilson"][1], "n": row["n"]}
    shift = {ls: {"n_adni4": r["n_adni4"], "n_older": r["n_older"],
                  "shift": r["shift_mean"]["accuracy"], "mixed": r["mixed_mean"]["accuracy"]}
             for ls, r in _load(os.path.join(v8, "phase_shift_bio.json")).items()}

    final = next(c for c in privacy if c["label_set"] == "cn-mci-ad" and c["target_epsilon"] == 1.0)
    return {
        "cohort": "v8",
        "conditions": centralised + with_bio + federated + privacy,
        "attack_auroc": {k: mean(v) for k, v in attacks.items()},
        "dimension_law": _load(os.path.join(v8, "metrics_dp27", "dimension_law.json"))["rows"],
        "dp_config": {**{k: run[k] for k in ("clip_norm", "rounds", "local_epochs",
                                             "subject_sample_rate", "delta", "accountant")},
                      "prox_mu": fedavg["prox_mu"]},
        "partition": {"n_clients": run["partition"]["n_clients"],
                      "n_sites": sum(c["n_sites"] for c in clients), "clients": clients},
        "by_sex": by_sex,
        "shift": shift,
        "history": {**previous["history"],
                    "v8": {"accuracy": final["accuracy"], "n_subjects": final["n_subjects"]}},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="export the site's data file from the v8 results")
    parser.add_argument("--root", default=".")
    parser.add_argument("--summary", default="results_v8_summary")
    parser.add_argument("--out", default=os.path.join("..", "ui", "public", "data", "site.json"))
    args = parser.parse_args(argv)
    previous = _load(args.out)  # earlier cohorts' headlines for the audit trail
    data = build(args.root, args.summary, previous)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
    print(f"wrote {args.out}: {len(data['conditions'])} conditions, "
          f"{data['partition']['n_clients']} clients, cohort {data['history']['v8']['n_subjects']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
