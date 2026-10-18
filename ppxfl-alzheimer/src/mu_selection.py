"""
mu_selection.py — choose the FedProx strength per client partition, honestly.

The paper's FedProx runs use one mu, chosen on the real-site partition and on
the folds it is reported on.  Here every partition chooses its own mu from the
grid on one seed and is scored on the other seeds, so selection and evaluation
never see the same split.  If the simulated partitions choose a different mu
from the real one, a benchmark built on them would tune the federated optimiser
for the wrong deployment.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

RUN = re.compile(r"fed_(?P<partition>[a-z]+)_(?P<task>[a-z-]+)_(?P<split>subject|site)_s(?P<seed>\d+)\.json$")


def load_runs(directories: dict[str, str]) -> dict:
    """``{(partition, task, split): {mu: {seed: accuracy}}}`` from fed_*.json runs."""
    runs: dict = defaultdict(lambda: defaultdict(dict))
    for mu, directory in directories.items():
        for path in glob.glob(os.path.join(directory, "fed_*.json")):
            match = RUN.search(os.path.basename(path))
            if not match:
                continue
            with open(path, encoding="utf-8") as handle:
                accuracy = json.load(handle)["overall"]["accuracy"]
            key = (match["partition"], match["task"], match["split"])
            runs[key][mu][int(match["seed"])] = accuracy
    return runs


def _mean(values):
    return sum(values) / len(values) if values else None


def _held_out(seeds: dict, selection_seed: int) -> list[float]:
    return [accuracy for seed, accuracy in seeds.items() if seed != selection_seed]


def select(runs: dict, selection_seed: int = 42) -> list[dict]:
    """Per configuration: the mu best on the selection seed, scored on the rest."""
    rows = []
    for (partition, task, split), by_mu in sorted(runs.items()):
        # ties go to the smaller mu, i.e. towards plain FedAvg
        chosen = max(by_mu, key=lambda mu, b=by_mu: (b[mu].get(selection_seed, -1.0), -float(mu)))
        rows.append({
            "partition": partition, "task": task, "split": split,
            "chosen_mu": float(chosen),
            "held_out_chosen": _mean(_held_out(by_mu[chosen], selection_seed)),
            "held_out_fedavg": _mean(_held_out(by_mu["0"], selection_seed)) if "0" in by_mu else None,
            "per_mu_mean": {mu: _mean(list(seeds.values())) for mu, seeds in by_mu.items()},
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FedProx mu chosen per partition")
    parser.add_argument("--run", nargs="+", required=True, metavar="MU=DIR",
                        help="directory of fed_*.json runs for each mu")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join("results_v7", "mu_selection.json"))
    args = parser.parse_args(argv)
    directories = dict(spec.split("=", 1) for spec in args.run)
    rows = select(load_runs(directories), args.selection_seed)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"selection_seed": args.selection_seed, "rows": rows}, handle, indent=2)
    for row in rows:
        print(f"  {row['partition']:9s} {row['task']:9s} {row['split']:7s} mu={row['chosen_mu']:g} "
              f"held-out {row['held_out_chosen']:.3f} vs FedAvg {row['held_out_fedavg']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
