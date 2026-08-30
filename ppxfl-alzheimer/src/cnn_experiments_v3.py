"""
cnn_experiments_v3.py — the convolutional arm of the study.

Two things need a trained network rather than a closed-form argument:

* the claim that anatomical standardisation, not model capacity, is what the
  earlier cohort was missing — which only holds if the *same* architecture
  family improves on the corrected input;
* the high-dimensional end of the privacy analysis, where the noise-to-signal
  ratio predicted from ``sqrt(d)`` should be visible as an actual collapse onto
  the majority class rather than only as a number.

Subject-level aggregation of slice probabilities keeps the unit of evaluation
the same as in the region-feature arm, so the two are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .cnn_v3 import count_parameters, get_backbone, train_one_model
from .evaluate_v3 import LABEL_SETS, _bootstrap, _metrics
from .splits_v3 import make_folds


def load_slices(path: str):
    data = np.load(path, allow_pickle=True)
    return (
        data["X"],
        data["y"].astype(int),
        list(data["subjects"]),
        list(data["sites"]),
        list(data["levels"]),
    )


def run_configuration(stacks, labels, sites, label_set: str, split_scheme: str,
                      backbone: str, head_only: bool, n_splits: int,
                      epochs: int, seed: int) -> dict:
    classes = LABEL_SETS[label_set]["classes"]
    keep = np.isin(labels, classes)
    index = np.flatnonzero(keep)
    remap = {c: i for i, c in enumerate(classes)}
    local_labels = np.zeros(len(labels), dtype=int)
    local_labels[index] = [remap[v] for v in labels[index]]
    subset_labels = local_labels[index]
    subset_sites = [sites[i] for i in index]
    n_classes = len(classes)

    folds = make_folds(split_scheme, subset_labels, subset_sites,
                       n_splits=n_splits, seed=seed)
    scores = np.zeros((len(index), n_classes))
    for fold in folds:
        probability, _ = train_one_model(
            stacks, local_labels, index[fold.train], index[fold.test],
            backbone=backbone, head_only=head_only, epochs=epochs, seed=seed,
        )
        scores[fold.test] = probability[index[fold.test]][:, :n_classes]

    predictions = np.argmax(scores, axis=1)
    overall = _metrics(subset_labels, predictions, scores, n_classes)
    overall.update(_bootstrap(subset_labels, predictions, scores, n_classes, seed=seed))
    counts = np.bincount(subset_labels, minlength=n_classes)

    total, trainable = count_parameters(
        get_backbone(backbone, n_classes, pretrained=False, head_only=head_only)
    )
    return {
        "model": f"{backbone}{'_head' if head_only else '_full'}",
        "label_set": label_set,
        "class_names": list(LABEL_SETS[label_set]["names"]),
        "split_scheme": split_scheme,
        "seed": seed,
        "epochs": epochs,
        "n_subjects": int(len(index)),
        "chance_accuracy": float(counts.max() / counts.sum()),
        "uniform_chance": 1.0 / n_classes,
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "overall": overall,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="2.5D CNN experiments, v3 cohort")
    parser.add_argument("--slices", default=os.path.join("data", "mni2mm", "slices.npz"))
    parser.add_argument("--out-dir", default=os.path.join("results_v3", "metrics"))
    parser.add_argument("--backbones", nargs="+", default=["resnet18", "resnet50"])
    parser.add_argument("--label-sets", nargs="+", default=["cn-mci-ad", "cn-ad"])
    parser.add_argument("--schemes", nargs="+", default=["subject"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--head-only", action="store_true",
                        help="freeze the backbone and train only the classifier head")
    args = parser.parse_args(argv)

    stacks, labels, subjects, sites, levels = load_slices(args.slices)
    print(f"{stacks.shape[0]} subjects, {stacks.shape[1]} MNI levels: "
          f"{', '.join(levels)}")
    os.makedirs(args.out_dir, exist_ok=True)

    summary = []
    for label_set in args.label_sets:
        for split_scheme in args.schemes:
            for backbone in args.backbones:
                for seed in args.seeds:
                    payload = run_configuration(
                        stacks, labels, sites, label_set, split_scheme, backbone,
                        args.head_only, args.n_splits, args.epochs, seed
                    )
                    tag = (f"cnn_{payload['model']}_{label_set}_"
                           f"{split_scheme}_s{seed}")
                    with open(os.path.join(args.out_dir, f"{tag}.json"), "w",
                              encoding="utf-8") as handle:
                        json.dump(payload, handle, indent=2)
                    summary.append({"tag": tag, **payload["overall"]})
                    print(f"  {tag}: acc={payload['overall']['accuracy']:.3f} "
                          f"bal={payload['overall']['balanced_accuracy']:.3f} "
                          f"F1={payload['overall']['macro_f1']:.3f} "
                          f"(chance {payload['chance_accuracy']:.3f}, "
                          f"{payload['trainable_parameters']:,} trainable)",
                          flush=True)

    with open(os.path.join(args.out_dir, "cnn_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"runs": summary}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
