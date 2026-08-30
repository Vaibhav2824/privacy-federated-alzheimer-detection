"""
export_ui_v3.py — render the v3 summary into the contract the dashboard reads.

The dashboard's ``ResultsSummary`` carries a single ``chance_accuracy``, so a
page that mixed the three-class conditions with the binary ones would draw a
34% chance line under a 79.7% CN-versus-AD row and invite exactly the wrong
reading.  Only the three-class conditions are exported for that reason, and the
cohort string says so.

Nothing is written into the page by hand: the dashboard renders whatever this
file produces, so it cannot drift from the recorded runs.
"""

from __future__ import annotations

import argparse
import json
import os

# The dashboard's method identifiers, which predate the v3 families.
METHOD_BY_FAMILY = {
    "centralised": "centralised",
    "federated": "fedavg",
    "privacy": "dpfedavg_userlevel",
}

# Perturbed dimension is only meaningful for the private arm; for the others the
# dashboard shows a dash.
PRIVATE_FAMILY = "privacy"


def condition_name(entry: dict) -> str:
    """A human-readable label for one condition."""
    parts = [entry["split_scheme"].replace("_", " ")]
    if entry["family"] == "federated":
        parts.append(f"{entry['client_scheme']} clients")
    if entry.get("target_epsilon") is not None:
        parts.append(f"eps {entry['target_epsilon']:g}")
    return ", ".join(parts)


def to_dashboard_condition(entry: dict) -> dict:
    accuracy = entry["accuracy"]
    f1 = entry["macro_f1"]
    auroc = entry["macro_auroc"]
    accounted = entry.get("accounted_epsilon") or {}
    return {
        "condition": condition_name(entry),
        "model": entry["model"],
        "method": METHOD_BY_FAMILY.get(entry["family"], entry["family"]),
        "dp_scope": "subject" if entry["family"] == PRIVATE_FAMILY else None,
        "epsilon": entry.get("target_epsilon"),
        "n_runs": entry["n_runs"],
        "folds": [],
        "seeds": [],
        "tags": [
            entry["family"], entry["model"], entry["label_set"],
            entry["split_scheme"], entry["client_scheme"],
        ],
        "accuracy_mean": accuracy["mean"],
        "accuracy_std": accuracy["std"],
        "f1_macro_mean": f1["mean"],
        "f1_macro_std": f1["std"],
        "auroc_macro_mean": auroc["mean"],
        "auroc_macro_std": auroc["std"],
        # Per-class precision and recall are not aggregated in v3; the dashboard
        # renders a dash rather than a number it cannot source.
        "precision_macro_mean": None,
        "recall_macro_mean": None,
        "actual_epsilon_mean": accounted.get("mean"),
        "perturbed_params": (
            entry.get("perturbed_dimension")
            if entry["family"] == PRIVATE_FAMILY else None
        ),
    }


def build(summary: dict, label_set: str = "cn-mci-ad") -> dict:
    selected = [c for c in summary["conditions"] if c["label_set"] == label_set]
    if not selected:
        raise ValueError(f"no conditions for label set {label_set!r}")
    conditions = [
        to_dashboard_condition(entry) for entry in selected
        if entry["accuracy"]["mean"] is not None
    ]
    conditions.sort(key=lambda c: (-(c["accuracy_mean"] or 0.0), c["condition"]))
    return {
        "cohort": (
            f"{selected[0]['n_subjects']} ADNI subjects, MNI152-registered, "
            "CN / MCI / AD, subject- and site-disjoint folds"
        ),
        "chance_accuracy": selected[0]["chance_accuracy"],
        "n_conditions": len(conditions),
        "conditions": conditions,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="export the dashboard summary")
    parser.add_argument("--summary",
                        default=os.path.join("results_v3", "results_summary_v3.json"))
    parser.add_argument("--out",
                        default=os.path.join("..", "ui", "public", "data",
                                             "results_summary.json"))
    parser.add_argument("--label-set", default="cn-mci-ad")
    args = parser.parse_args(argv)

    with open(args.summary, encoding="utf-8") as handle:
        summary = json.load(handle)
    payload = build(summary, args.label_set)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {args.out}: {payload['n_conditions']} conditions, "
          f"chance {payload['chance_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
