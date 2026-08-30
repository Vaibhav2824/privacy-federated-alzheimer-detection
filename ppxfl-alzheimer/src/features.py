"""
features.py — atlas region morphometry from MNI-registered T1 volumes.

Once every subject sits on the same MNI grid, a voxel index means the same
anatomy in all of them, so anatomy can be summarised directly instead of being
learned from scratch by a 23.5M-parameter network on 299 subjects.

Each volume is segmented into CSF / grey matter / white matter with a
three-component Gaussian mixture over brain-mask intensities, and the grey
matter map is integrated over Harvard-Oxford regions.  Region values are
expressed as a fraction of total intracranial volume, which removes head size —
the single largest nuisance in raw region volumes — from the comparison.

The resulting representation is order 10^2 features rather than 10^7 voxels.
That matters twice over: it is the regime where a few hundred subjects can
actually support estimation, and it is the regime where subject-level
differential privacy is affordable, because the Gaussian mechanism's noise norm
grows as the square root of the perturbed dimension.
"""

from __future__ import annotations

import json
import os

import numpy as np

# Regions with a documented role in Alzheimer's structural change, used for the
# medial-temporal summary and for the explainability analysis.
MEDIAL_TEMPORAL_REGIONS = (
    "Left Hippocampus",
    "Right Hippocampus",
    "Left Amygdala",
    "Right Amygdala",
)
VENTRICLE_REGIONS = ("Left Lateral Ventricle", "Right Lateral Ventricle")


def tissue_posteriors(volume: np.ndarray, mask: np.ndarray, seed: int = 0,
                      fit_samples: int = 20_000):
    """CSF / GM / WM posterior maps from a three-component intensity mixture.

    Components are sorted by mean intensity, which on a T1 orders them
    CSF < GM < WM.  Only voxels inside the brain mask take part; everything
    else is zero in all three maps.

    The mixture is fitted on a random subsample of the in-mask voxels and then
    applied to all of them.  A three-component mixture over a scalar has eight
    free parameters, so 20,000 draws estimate it as well as 235,000 do, and the
    subsample turns a per-subject fit that dominates the feature stage into one
    that does not.
    """
    from sklearn.mixture import GaussianMixture

    values = volume[mask].astype(np.float64).reshape(-1, 1)
    if values.size == 0 or values.std() == 0:
        zeros = np.zeros(volume.shape, dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy()

    if values.shape[0] > fit_samples:
        rng = np.random.default_rng(seed)
        sample = values[rng.choice(values.shape[0], fit_samples, replace=False)]
    else:
        sample = values

    model = GaussianMixture(
        n_components=3, covariance_type="full", random_state=seed, max_iter=200
    ).fit(sample)
    order = np.argsort(model.means_.ravel())
    posteriors = model.predict_proba(values)[:, order]

    maps = []
    for component in range(3):
        out = np.zeros(volume.shape, dtype=np.float32)
        out[mask] = posteriors[:, component].astype(np.float32)
        maps.append(out)
    return tuple(maps)


def load_atlas(template, labels_only: bool = False):
    """Harvard-Oxford cortical + subcortical labels on the template grid."""
    from nilearn import datasets
    from nilearn.image import resample_to_img

    names: list[str] = []
    label_volume = None
    offset = 0
    for atlas_name in ("cort-maxprob-thr25-2mm", "sub-maxprob-thr25-2mm"):
        atlas = datasets.fetch_atlas_harvard_oxford(atlas_name)
        if labels_only:
            names.extend(
                f"{label}" for label in atlas.labels[1:]
            )
            continue
        resampled = resample_to_img(
            atlas.maps, template, interpolation="nearest",
            copy_header=True, force_resample=True,
        )
        data = np.asarray(resampled.dataobj).astype(np.int32)
        if label_volume is None:
            label_volume = np.zeros(data.shape, dtype=np.int32)
        # Subcortical labels are offset so the two atlases do not collide.
        shifted = np.where(data > 0, data + offset, 0)
        label_volume = np.where(label_volume == 0, shifted, label_volume)
        names.extend(atlas.labels[1:])
        offset += len(atlas.labels) - 1
    return label_volume, names


def region_features(volume: np.ndarray, mask: np.ndarray, label_volume: np.ndarray,
                    names: list[str], seed: int = 0):
    """Per-region grey-matter and CSF fractions, normalised by brain volume.

    Returns ``(vector, feature_names)``.  Each region contributes its grey
    matter volume and its CSF volume as fractions of total brain volume; two
    global summaries (overall grey matter fraction and overall CSF fraction)
    are appended.
    """
    csf, grey, white = tissue_posteriors(volume, mask, seed=seed)
    brain_volume = float(grey.sum() + white.sum() + csf.sum())
    if brain_volume <= 0:
        brain_volume = 1.0

    values, feature_names = [], []
    for index, name in enumerate(names, start=1):
        region = label_volume == index
        if not region.any():
            values.extend([0.0, 0.0])
        else:
            values.append(float(grey[region].sum()) / brain_volume)
            values.append(float(csf[region].sum()) / brain_volume)
        feature_names.append(f"gm::{name}")
        feature_names.append(f"csf::{name}")

    values.append(float(grey.sum()) / brain_volume)
    values.append(float(csf.sum()) / brain_volume)
    feature_names.append("gm::global")
    feature_names.append("csf::global")
    return np.asarray(values, dtype=np.float32), feature_names


def build_feature_matrix(manifest_rows, volume_dir: str, template, mask,
                         seed: int = 0, progress: bool = True):
    """Feature matrix for every subject that passed registration QC."""
    label_volume, names = load_atlas(template)
    matrix, subjects, labels, sites, feature_names = [], [], [], [], None
    for index, row in enumerate(manifest_rows, start=1):
        volume = np.load(os.path.join(volume_dir, f"{row['subject_id']}.npy")).astype(np.float32)
        vector, feature_names = region_features(
            volume, mask, label_volume, names, seed=seed
        )
        matrix.append(vector)
        subjects.append(row["subject_id"])
        labels.append(int(row["label"]))
        sites.append(row["site_id"])
        if progress and index % 25 == 0:
            print(f"  features {index}/{len(manifest_rows)}", flush=True)
    return (
        np.asarray(matrix, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        subjects,
        sites,
        feature_names,
    )


def save_features(path: str, matrix, labels, subjects, sites, feature_names) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        X=matrix,
        y=labels,
        subjects=np.asarray(subjects),
        sites=np.asarray(sites),
        feature_names=np.asarray(feature_names),
    )
    meta = {
        "n_subjects": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "medial_temporal_features": [
            f"{prefix}::{name}"
            for name in MEDIAL_TEMPORAL_REGIONS for prefix in ("gm", "csf")
            if f"{prefix}::{name}" in feature_names
        ],
    }
    with open(os.path.splitext(path)[0] + "_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def main(argv=None) -> int:
    import argparse
    import csv

    from .preprocess_v3 import _templates

    parser = argparse.ArgumentParser(description="atlas morphometry features")
    parser.add_argument("--data-dir", default=os.path.join("data", "mni2mm"))
    parser.add_argument("--out", default=os.path.join("data", "mni2mm", "features.npz"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    template, mask = _templates()
    with open(os.path.join(args.data_dir, "manifest.csv"), encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["qc_pass"] == "1"]
    print(f"building features for {len(rows)} QC-passing subjects")

    matrix, labels, subjects, sites, names = build_feature_matrix(
        rows, os.path.join(args.data_dir, "vol"), template, mask, seed=args.seed
    )
    save_features(args.out, matrix, labels, subjects, sites, names)
    print(f"wrote {args.out} with shape {matrix.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
