"""
figures_v3.py — publication figures for the rebuilt pipeline.

Every accuracy axis carries its chance band, because several configurations in
this study sit close to chance and a bar chart without one invites the reader to
over-read a few points. The palette is colour-blind safe and figures render at
300 dpi.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

PALETTE = {
    "centralised": "#0072B2",
    "federated": "#D55E00",
    "private": "#009E73",
    "reference": "#CC79A7",
    "grey": "#666666",
    "chance": "#BBBBBB",
}
DPI = 300


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 110,
    })
    return plt


def _load_all(results_dir: str, pattern: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, pattern))):
        with open(path, encoding="utf-8") as handle:
            out.append(json.load(handle))
    return out


def figure_preprocessing(volume_dir: str, template, out_path: str, n: int = 4) -> None:
    """Registered subjects beside the template, as the QC the reader can check."""
    plt = _style()
    paths = sorted(glob.glob(os.path.join(volume_dir, "*.npy")))[:n]
    template_data = np.asarray(template.dataobj, dtype=np.float32)
    rows = [("MNI152", template_data)] + [
        (os.path.basename(p)[:-4], np.load(p).astype(np.float32)) for p in paths
    ]
    figure, axes = plt.subplots(len(rows), 3, figsize=(6.4, 1.9 * len(rows)))
    for row, (name, volume) in enumerate(rows):
        views = [volume[:, :, volume.shape[2] // 2],
                 volume[:, volume.shape[1] // 2, :],
                 volume[volume.shape[0] // 2, :, :]]
        for column, view in enumerate(views):
            axis = axes[row, column]
            axis.imshow(np.rot90(view), cmap="gray")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            if row == 0:
                axis.set_title(["axial", "coronal", "sagittal"][column], fontsize=9)
        axes[row, 0].set_ylabel(name, fontsize=8)
    figure.suptitle("Every subject on the MNI152 2 mm grid", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def legacy_slice(nifti_path: str, target: int = 224):
    """Reproduce the v1/v2 slice rule exactly, for the comparison figure.

    The rule picked the longest array axis and took the middle slice of it,
    with no reorientation and no registration.  Reproducing it here rather than
    describing it lets the reader see what it extracts.
    """
    import nibabel as nib
    from scipy import ndimage as ndi

    data = np.asarray(nib.load(nifti_path).dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    axis = int(np.argmax(data.shape))
    index = data.shape[axis] // 2
    plane = np.take(data, index, axis=axis)
    if plane.std() > 0:
        plane = np.clip((plane - plane.mean()) / plane.std(), -3, 3)
        plane = (plane - plane.min()) / (plane.max() - plane.min() + 1e-8)
    zoom = (target / plane.shape[0], target / plane.shape[1])
    return ndi.zoom(plane, zoom, order=1)[:target, :target], axis


def figure_pipeline_comparison(manifest_rows, volume_dir: str, affine,
                               out_path: str, n: int = 4) -> None:
    """The old slice rule beside the standardised slice, for the same subjects.

    Subjects are chosen so that the old rule picks a different array axis for
    different subjects, which is the defect the figure exists to show.
    """
    plt = _style()
    from .slices_v3 import extract_slices

    chosen, axes_seen = [], set()
    for row in manifest_rows:
        if row.get("qc_pass") != "1":
            continue
        path = os.path.join(volume_dir, f"{row['subject_id']}.npy")
        if not os.path.exists(path) or not os.path.exists(row["source_path"]):
            continue
        try:
            legacy, axis = legacy_slice(row["source_path"])
        except Exception:
            continue
        if axis in axes_seen and len(chosen) >= 2:
            continue
        axes_seen.add(axis)
        chosen.append((row, legacy, axis, np.load(path).astype(np.float32)))
        if len(chosen) >= n:
            break
    if not chosen:
        return

    figure, axes = plt.subplots(2, len(chosen),
                               figsize=(2.0 * len(chosen), 4.4), squeeze=False)
    for column, (row, legacy, axis, volume) in enumerate(chosen):
        stack, descriptions = extract_slices(volume, affine, size=224)
        # The axial level through the medial temporal lobe.
        level = descriptions.index("z=-18") if "z=-18" in descriptions else 0
        for panel, image, title in (
            (axes[0][column], legacy, f"axis {axis}"),
            (axes[1][column], stack[level][1], descriptions[level]),
        ):
            panel.imshow(np.rot90(image), cmap="gray")
            panel.set_xticks([])
            panel.set_yticks([])
            panel.grid(False)
            panel.set_title(f"{row['klass']}, {title}", fontsize=7)
    axes[0][0].set_ylabel("previous rule\n(longest axis)", fontsize=8)
    axes[1][0].set_ylabel("standardised\n(fixed MNI level)", fontsize=8)
    figure.suptitle("The same subjects under both slice rules", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def figure_qc(manifest_rows, out_path: str, threshold: float) -> None:
    """Distribution of registration quality, with the exclusion rule marked."""
    plt = _style()
    scores = np.asarray([float(r["qc_correlation"]) for r in manifest_rows])
    figure, axis = plt.subplots(figsize=(5.2, 3.0))
    axis.hist(scores, bins=30, color=PALETTE["centralised"], alpha=0.85)
    axis.axvline(threshold, color=PALETTE["reference"], linestyle="--",
                 label=f"exclusion threshold r = {threshold}")
    passed = int((scores >= threshold).sum())
    axis.set_xlabel("registration correlation with MNI152 template")
    axis.set_ylabel("subjects")
    axis.set_title(f"Registration QC: {passed}/{len(scores)} subjects retained")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def figure_partition(partition: dict, out_path: str) -> None:
    """Per-client class composition of the real ADNI site partition."""
    plt = _style()
    clients = partition["clients"]
    labels = ["CN", "MCI", "AD"]
    colours = [PALETTE["centralised"], PALETTE["federated"], PALETTE["private"]]
    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    bottom = np.zeros(len(clients))
    positions = np.arange(len(clients))
    for index, (name, colour) in enumerate(zip(labels, colours)):
        heights = np.asarray([c["class_counts"][str(index)] for c in clients], float)
        axis.bar(positions, heights, bottom=bottom, color=colour, label=name)
        bottom += heights
    axis.set_xticks(positions)
    axis.set_xticklabels(
        ["+".join(c["sites"]) if len(c["sites"]) <= 2 else f"{len(c['sites'])} sites"
         for c in clients],
        rotation=60, ha="right", fontsize=7,
    )
    missing = partition["clients_missing_a_class"]
    axis.set_ylabel("subjects")
    axis.set_title(
        f"Natural ADNI site partition: {len(clients)} clients, "
        f"{missing} missing at least one class"
    )
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def figure_accuracy(runs: list[dict], out_path: str, title: str) -> None:
    """Accuracy and macro F1 per configuration, with the chance band drawn."""
    plt = _style()
    if not runs:
        return
    names = [r["tag"] for r in runs]
    accuracy = [r["accuracy"] for r in runs]
    macro_f1 = [r["macro_f1"] for r in runs]
    chance = [r.get("chance", r.get("chance_accuracy", 1 / 3)) for r in runs]

    positions = np.arange(len(runs))
    figure, axis = plt.subplots(figsize=(max(6.0, 0.5 * len(runs)), 3.4))
    axis.bar(positions - 0.2, accuracy, width=0.4, label="accuracy",
             color=PALETTE["centralised"])
    axis.bar(positions + 0.2, macro_f1, width=0.4, label="macro F1",
             color=PALETTE["federated"])
    axis.plot(positions, chance, color=PALETTE["chance"], linestyle="--",
              marker="_", label="majority-class rate")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=70, ha="right", fontsize=6)
    axis.set_ylim(0, 1)
    axis.set_ylabel("score")
    axis.set_title(title)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def figure_dimension_law(law: dict, out_path: str) -> None:
    """Noise-to-signal against perturbed dimension, with the sqrt(d) reference."""
    plt = _style()
    rows = law["rows"]
    epsilons = sorted({r["target_epsilon"] for r in rows})
    figure, axis = plt.subplots(figsize=(5.4, 3.4))
    markers = ["o", "s", "^", "D", "v"]
    for marker, epsilon in zip(markers, epsilons):
        subset = sorted((r for r in rows if r["target_epsilon"] == epsilon),
                        key=lambda r: r["dimension"])
        axis.plot([r["dimension"] for r in subset],
                  [r["worst_case_ratio"] for r in subset],
                  marker=marker, label=f"$\\varepsilon$ = {epsilon:g}")
    axis.axhline(1.0, color=PALETTE["grey"], linestyle=":",
                 label="noise equals signal")

    # Name the three configurations, so the axis reads as a design choice
    # rather than as an abstract dimension.
    labels = {
        "roi_logreg": "MNI region model",
        "resnet50_head": "ResNet50 head",
        "resnet50_full": "ResNet50, full",
    }
    seen = {}
    for row in rows:
        seen.setdefault(row["model"], row["dimension"])
    top = max(r["worst_case_ratio"] for r in rows)
    for model, dimension in seen.items():
        axis.axvline(dimension, color=PALETTE["grey"], alpha=0.15, linewidth=1)
        axis.annotate(labels.get(model, model), xy=(dimension, top),
                      xytext=(0, 4), textcoords="offset points",
                      rotation=90, ha="center", va="bottom", fontsize=7,
                      color=PALETTE["grey"])

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("perturbed parameters $d$")
    axis.set_ylabel("worst-case noise / signal norm")
    axis.set_title("Subject-level DP cost is set by model dimension")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def figure_privacy_utility(runs: list[dict], out_path: str) -> None:
    """Utility and explanation agreement against the privacy budget."""
    plt = _style()
    private = [r for r in runs if r.get("target_epsilon")]
    baseline = [r for r in runs if not r.get("target_epsilon")]
    if not private:
        return
    epsilons = sorted({r["target_epsilon"] for r in private})

    def mean_of(key, epsilon):
        values = [r[key] for r in private
                  if r["target_epsilon"] == epsilon and r.get(key) is not None
                  and not np.isnan(r[key])]
        return float(np.mean(values)) if values else np.nan

    figure, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for key, colour, label in (("accuracy", PALETTE["centralised"], "accuracy"),
                               ("macro_f1", PALETTE["federated"], "macro F1")):
        left.plot(epsilons, [mean_of(key, e) for e in epsilons], marker="o",
                  color=colour, label=label)
        if baseline:
            reference = float(np.mean([b[key] for b in baseline]))
            left.axhline(reference, color=colour, linestyle="--", alpha=0.5)
    if private:
        left.axhline(private[0].get("chance", 1 / 3), color=PALETTE["chance"],
                     linestyle=":", label="majority-class rate")
    left.set_xscale("log")
    left.set_xlabel("$\\varepsilon$ (subject-level)")
    left.set_ylabel("score")
    left.set_title("Utility against privacy budget")
    left.legend(frameon=False, fontsize=8)

    right.plot(epsilons, [mean_of("explainability", e) for e in epsilons],
               marker="s", color=PALETTE["private"])
    right.axhline(1.0, color=PALETTE["grey"], linestyle=":")
    right.set_xscale("log")
    right.set_ylim(-0.1, 1.05)
    right.set_xlabel("$\\varepsilon$ (subject-level)")
    right.set_ylabel("Spearman agreement with non-private model")
    right.set_title("Explanation fidelity against privacy budget")

    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="figures for the rebuilt pipeline")
    parser.add_argument("--results-dir", default=os.path.join("results_v3", "metrics"))
    parser.add_argument("--data-dir", default=os.path.join("data", "mni2mm"))
    parser.add_argument("--out-dir", default=os.path.join("..", "figures", "v3"))
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    import csv

    from .preprocess_v3 import QC_MIN_CORRELATION, _templates

    template, _ = _templates()
    manifest_path = os.path.join(args.data_dir, "manifest.csv")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        figure_qc(rows, os.path.join(args.out_dir, "registration_qc.png"),
                  QC_MIN_CORRELATION)
        figure_preprocessing(os.path.join(args.data_dir, "vol"), template,
                             os.path.join(args.out_dir, "mni_registration.png"))
        figure_pipeline_comparison(
            rows, os.path.join(args.data_dir, "vol"), template.affine,
            os.path.join(args.out_dir, "slice_rule_comparison.png"),
        )

    summary_path = os.path.join(args.results_dir, "centralised_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            figure_accuracy(json.load(handle),
                            os.path.join(args.out_dir, "centralised_accuracy.png"),
                            "Centralised baselines, subject- and site-disjoint")

    federated_path = os.path.join(args.results_dir, "federated_summary.json")
    if os.path.exists(federated_path):
        with open(federated_path, encoding="utf-8") as handle:
            runs = json.load(handle)["runs"]
        figure_accuracy(runs, os.path.join(args.out_dir, "federated_accuracy.png"),
                        "Federated averaging by client partition")

    for candidate in sorted(glob.glob(os.path.join(args.results_dir, "fed_natural_*.json"))):
        with open(candidate, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("partition"):
            figure_partition(payload["partition"],
                             os.path.join(args.out_dir, "site_partition.png"))
            break

    law_path = os.path.join(args.results_dir, "dimension_law.json")
    if os.path.exists(law_path):
        with open(law_path, encoding="utf-8") as handle:
            figure_dimension_law(json.load(handle),
                                 os.path.join(args.out_dir, "dimension_law.png"))

    privacy_path = os.path.join(args.results_dir, "privacy_summary.json")
    if os.path.exists(privacy_path):
        with open(privacy_path, encoding="utf-8") as handle:
            figure_privacy_utility(json.load(handle)["runs"],
                                   os.path.join(args.out_dir, "privacy_utility.png"))

    print(f"figures written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
