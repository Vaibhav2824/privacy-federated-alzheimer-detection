"""
vbm_features.py — region volumes from a proper VBM pipeline (deepmriprep).

``features.py`` segments affine-registered volumes with an intensity-only
mixture and integrates over a fixed Harvard-Oxford atlas.  That leaves the
hippocampus label on different anatomy in different subjects, and voxels the
skull strip removed inside the template mask are counted as CSF.

deepmriprep (CAT12 in PyTorch, by the deepbet authors) segments each brain
with a trained network, warps nonlinearly to the CAT12 template and warps the
Neuromorphometrics atlas back, so every region volume is measured on the
subject's own anatomy.  Each region contributes the tissue CAT12 assigns it
(grey matter for cortex and nuclei, CSF for ventricles), divided by total
intracranial volume.  Same ~140-feature regime, so the DP dimension argument
is unchanged.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .features import save_features

ATLAS = "neuromorphometrics"
TISSUE = (("Vgm", "gmv_mm3", "gm"), ("Vwm", "wmv_mm3", "wm"), ("Vcsf", "csfv_mm3", "csf"))


def _atlas_tissues() -> dict[int, tuple[str, str]]:
    """ROIid -> (column, prefix) for the tissue CAT12 measures in that region."""
    import deepmriprep

    path = os.path.join(os.path.dirname(deepmriprep.__file__), "data", "templates", f"{ATLAS}.csv")
    out = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            for flag, column, prefix in TISSUE:
                if row[flag] == "1":
                    out[int(row["ROIid"])] = (column, prefix)
                    break
    return out


def run(rows, bet_dir: str, out_dir: str) -> None:
    """Run deepmriprep on every stripped brain not already processed."""
    from deepmriprep import run_preprocess

    todo = [os.path.join(bet_dir, f"{r['subject_id']}.nii.gz") for r in rows
            if not os.path.exists(os.path.join(out_dir, f"tiv{r['subject_id']}.csv"))]
    print(f"deepmriprep: {len(todo)} of {len(rows)} subjects to process", flush=True)
    if todo:
        run_preprocess(input_paths=todo, output_dir=out_dir, dir_format="flat",
                       outputs=[f"{ATLAS}_volumes", "tiv", "warp_mse"])


def subject_vector(out_dir: str, subject: str, tissues: dict):
    with open(os.path.join(out_dir, f"tiv{subject}.csv"), encoding="utf-8") as handle:
        tiv_mm3 = float(list(csv.DictReader(handle))[0]["tiv_cm3"]) * 1000.0
    values, names = [], []
    with open(os.path.join(out_dir, f"{ATLAS}_volumes{subject}.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            column, prefix = tissues[int(row["ROIid"])]
            values.append(float(row[column]) / tiv_mm3)
            names.append(f"{prefix}::{row['ROIname']}")
    return np.asarray(values, dtype=np.float32), names


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="deepmriprep region morphometry")
    parser.add_argument("--data-dir", default=os.path.join("data", "mni2mm"))
    parser.add_argument("--out", default=os.path.join("data", "mni2mm", "features_vbm.npz"))
    args = parser.parse_args(argv)

    with open(os.path.join(args.data_dir, "manifest.csv"), encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["qc_pass"] == "1"]
    out_dir = os.path.join(args.data_dir, "dmp")
    run(rows, os.path.join(args.data_dir, "bet"), out_dir)

    tissues = _atlas_tissues()
    matrix, kept, names = [], [], None
    for row in rows:
        if not os.path.exists(os.path.join(out_dir, f"tiv{row['subject_id']}.csv")):
            print(f"  [skip] deepmriprep produced nothing for {row['subject_id']}")
            continue
        vector, names = subject_vector(out_dir, row["subject_id"], tissues)
        matrix.append(vector)
        kept.append(row)
    save_features(args.out, np.asarray(matrix, dtype=np.float32),
                  np.asarray([int(r["label"]) for r in kept], dtype=np.int64),
                  [r["subject_id"] for r in kept], [r["site_id"] for r in kept], names)
    print(f"wrote {args.out} with shape {len(matrix)}x{len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
