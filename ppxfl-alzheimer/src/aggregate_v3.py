"""
aggregate_v3.py — collapse the per-run result JSONs into one summary.

Each experimental cell is repeated over seeds, so the reportable quantity is a
mean and a spread across seeds rather than any single run.  Conditions are
keyed by the full specification — representation, label set, split scheme,
client partition, privacy budget — so nothing is aggregated across a difference
that matters.

The output feeds both the paper tables and the results dashboard, which is what
keeps a number in the prose and a number in a figure from drifting apart.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np

METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc")


def _summarise(values):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


def load_runs(results_dir: str) -> list[dict]:
    """Every per-run JSON, tagged with the family it belongs to."""
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        if name.endswith("_summary.json") or name == "dimension_law.json":
            continue
        # A results directory is written by several tools and can contain a
        # path that merely looks like a run: a directory whose name ends .json,
        # a partial write from an interrupted job.  One of those should not
        # abort aggregation of every real run beside it.
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            print(f"  [skip] {name}: {error}")
            continue
        if not isinstance(payload, dict) or "overall" not in payload:
            continue
        payload["_file"] = name
        if name.startswith("dp_"):
            payload["_family"] = "privacy"
        elif name.startswith("fed_"):
            payload["_family"] = "federated"
        else:
            payload["_family"] = "centralised"
        runs.append(payload)
    return runs


def condition_key(run: dict) -> tuple:
    return (
        run["_family"],
        run.get("model", "roi_logreg"),
        run["label_set"],
        run["split_scheme"],
        run.get("client_scheme", "none"),
        run.get("target_epsilon"),
    )


def aggregate(results_dir: str) -> dict:
    runs = load_runs(results_dir)
    grouped = collections.defaultdict(list)
    for run in runs:
        grouped[condition_key(run)].append(run)

    conditions = []
    for key, members in sorted(grouped.items(), key=lambda item: str(item[0])):
        family, model, label_set, split_scheme, client_scheme, epsilon = key
        entry = {
            "family": family,
            "model": model,
            "label_set": label_set,
            "split_scheme": split_scheme,
            "client_scheme": client_scheme,
            "target_epsilon": epsilon,
            "n_runs": len(members),
            "n_subjects": members[0]["n_subjects"],
            "chance_accuracy": members[0]["chance_accuracy"],
            "uniform_chance": members[0]["uniform_chance"],
        }
        for metric in METRIC_KEYS:
            entry[metric] = _summarise([m["overall"].get(metric) for m in members])
        if family == "privacy":
            entry["accounted_epsilon"] = _summarise(
                [m.get("epsilon") for m in members]
            )
            entry["noise_multiplier"] = members[0].get("noise_multiplier")
            entry["perturbed_dimension"] = members[0].get("perturbed_dimension")
            agreements = [
                m.get("explainability", {}).get("agreement_spearman") for m in members
            ]
            entry["explanation_agreement"] = _summarise(agreements)
            entry["medial_temporal_share"] = _summarise([
                m.get("explainability", {}).get("medial_temporal_share")
                for m in members
            ])
        conditions.append(entry)

    return {
        "results_dir": results_dir,
        "n_runs": len(runs),
        "n_conditions": len(conditions),
        "conditions": conditions,
    }


def best_of(summary: dict, family: str, label_set: str, split_scheme: str,
            metric: str = "balanced_accuracy"):
    """The best condition in a family, for the paper's headline sentences."""
    candidates = [
        c for c in summary["conditions"]
        if c["family"] == family and c["label_set"] == label_set
        and c["split_scheme"] == split_scheme and c[metric]["mean"] is not None
        and c.get("target_epsilon") is None
    ]
    return max(candidates, key=lambda c: c[metric]["mean"]) if candidates else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="aggregate v3 results")
    parser.add_argument("--results-dir", default=os.path.join("results_v3", "metrics"))
    parser.add_argument("--out", default=os.path.join("results_v3", "results_summary_v3.json"))
    args = parser.parse_args(argv)

    summary = aggregate(args.results_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"{summary['n_runs']} runs -> {summary['n_conditions']} conditions")
    for condition in summary["conditions"]:
        accuracy = condition["accuracy"]["mean"]
        balanced = condition["balanced_accuracy"]["mean"]
        if accuracy is None:
            continue
        epsilon = condition["target_epsilon"]
        label = f" eps={epsilon}" if epsilon else ""
        print(f"  {condition['family']:<12} {condition['model']:<10} "
              f"{condition['label_set']:<10} {condition['split_scheme']:<8} "
              f"{condition['client_scheme']:<10}{label:<9} "
              f"acc={accuracy:.3f} bal={balanced:.3f} "
              f"(chance {condition['chance_accuracy']:.3f}, n={condition['n_runs']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
