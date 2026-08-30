"""
run_v3.py — end-to-end driver for the rebuilt pipeline.

Stages run in order and each is skippable, because the expensive ones
(registration, slice extraction) are cached on disk and the cheap ones are
re-run constantly while the analysis is being written.

    python run_v3.py --stages merge-orientation preprocess features centralised
    python run_v3.py --stages federated privacy dimension-law
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

PYTHON = sys.executable
STAGES = (
    "merge-orientation",
    "handedness",
    "preprocess",
    "features",
    "slices",
    "centralised",
    "cnn",
    "federated",
    "privacy",
    "dimension-law",
    "aggregate",
    "figures",
)


def merge_orientation(data_dir: str, out_path: str) -> None:
    """Combine the per-group orientation fits into one table."""
    merged = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "orient_g*.json"))):
        with open(path, encoding="utf-8") as handle:
            merged.update(json.load(handle))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
    print(f"merged {len(merged)} orientation groups into {out_path}")
    for key, value in sorted(merged.items()):
        print(f"  {key}: perm={value['permutation']} flips={value['flips']} "
              f"det={value['determinant']} votes={value['votes']}/{value['sampled']} "
              f"scans={value['scans_in_group']}")


def call(module: str, *arguments: str) -> None:
    command = [PYTHON, "-u", "-m", module, *arguments]
    print("\n$ " + " ".join(command), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"stage failed: {module} ({result.returncode})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="rebuilt pipeline driver")
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--data-root", default="..")
    parser.add_argument("--data-dir", default=os.path.join("data", "mni2mm"))
    parser.add_argument("--results-dir", default=os.path.join("results_v3", "metrics"))
    args = parser.parse_args(argv)

    orientation_path = os.path.join("data", "orientation_table.json")
    features_path = os.path.join(args.data_dir, "features.npz")

    for stage in args.stages:
        if stage == "merge-orientation":
            merge_orientation("data", orientation_path)
        elif stage == "handedness":
            call("src.handedness", "--data-root", args.data_root,
                 "--table", orientation_path, "--population-fallback")
        elif stage == "preprocess":
            call("src.preprocess_v3", "--data-root", args.data_root,
                 "--out", args.data_dir, "--orientation-table", orientation_path)
        elif stage == "features":
            call("src.features", "--data-dir", args.data_dir, "--out", features_path)
        elif stage == "slices":
            call("src.slices_v3", "--data-dir", args.data_dir,
                 "--out", os.path.join(args.data_dir, "slices.npz"))
        elif stage == "centralised":
            call("src.evaluate_v3", "--features", features_path,
                 "--out-dir", args.results_dir)
        elif stage == "cnn":
            call("src.cnn_experiments_v3",
                 "--slices", os.path.join(args.data_dir, "slices.npz"),
                 "--out-dir", args.results_dir)
        elif stage == "federated":
            call("src.experiments_v3", "--features", features_path,
                 "--out-dir", args.results_dir, "federated")
        elif stage == "privacy":
            call("src.experiments_v3", "--features", features_path,
                 "--out-dir", args.results_dir, "privacy")
        elif stage == "dimension-law":
            call("src.experiments_v3", "--features", features_path,
                 "--out-dir", args.results_dir, "dimension-law")
        elif stage == "aggregate":
            call("src.aggregate_v3", "--results-dir", args.results_dir,
                 "--out", os.path.join(os.path.dirname(args.results_dir),
                                       "results_summary_v3.json"))
        elif stage == "figures":
            call("src.figures_v3", "--results-dir", args.results_dir,
                 "--data-dir", args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
