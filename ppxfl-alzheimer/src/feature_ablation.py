"""
feature_ablation.py — does a richer region representation survive privacy?

Runs the same feature sets through the centralised nested-CV logistic model and
through subject-level DP-FedAvg at a strong and a weak budget.  More regions
and tissues add signal, but they also add perturbed parameters, so the ablation
measures the dimension law on real features rather than only in closed form.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .evaluate_v3 import LABEL_SETS, load_features, run_configuration
from .experiments_v3 import run_federated_configuration
from .federated_v3 import FedConfig, calibrate_noise

SEEDS = (42, 123, 2024)


def _mean(values) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def ablate(X, y, sites, feature_names, label_set: str, epsilons, *, rounds: int = 60,
           local_epochs: int = 3, learning_rate: float = 0.5, sample_rate: float = 0.5,
           delta: float = 1e-3, n_splits: int = 5, seeds=SEEDS) -> dict:
    """Centralised and DP accuracy for one feature set and label set."""
    central = [run_configuration(X, y, sites, "logreg_cv", label_set, "subject",
                                 n_splits, seed)["overall"]["accuracy"] for seed in seeds]
    private = {}
    for epsilon in epsilons:
        noise = calibrate_noise(epsilon, sample_rate, rounds, delta, accountant="prv")
        private[f"{epsilon:g}"] = _mean([
            run_federated_configuration(
                X, y, sites, label_set, "subject", "natural",
                FedConfig(rounds=rounds, local_epochs=local_epochs,
                          learning_rate=learning_rate, noise_multiplier=noise,
                          subject_sample_rate=sample_rate, seed=seed, delta=delta,
                          accountant="prv"),
                n_splits, feature_names)["overall"]["accuracy"]
            for seed in seeds])
    n_classes = len(LABEL_SETS[label_set]["classes"])
    return {"n_features": int(X.shape[1]),
            "perturbed_dimension": int(X.shape[1] * n_classes + n_classes),
            "centralised": _mean(central), "private": private}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="feature-dimension ablation under DP")
    parser.add_argument("--features", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--label-sets", nargs="+", default=["cn-ad", "cn-mci-ad"])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 10.0])
    parser.add_argument("--out", default=os.path.join("results_v7", "metrics", "feature_ablation.json"))
    args = parser.parse_args(argv)

    rows = []
    for spec in args.features:
        name, path = spec.split("=", 1)
        X, y, _, sites, feature_names = load_features(path)
        for label_set in args.label_sets:
            row = {"features": name, "label_set": label_set,
                   **ablate(X, y, sites, feature_names, label_set, args.epsilons)}
            rows.append(row)
            print(f"  {name} {label_set}: d={row['perturbed_dimension']} "
                  f"central={row['centralised']['mean']:.3f} "
                  + " ".join(f"eps{e}={v['mean']:.3f}" for e, v in row["private"].items()),
                  flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
