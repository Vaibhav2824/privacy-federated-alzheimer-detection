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
# Extra atlases for the extended (feature-dimension ablation) set: cobra and
# julichbrain carry hippocampal subfields, entorhinal cortex and amygdala nuclei.
EXTRA_ATLASES = ("cobra", "hammers", "lpba40", "julichbrain")
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


def _region_rows(out_dir, atlas, subject):
    with open(os.path.join(out_dir, f"{atlas}_volumes{subject}.csv"), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _value(cell) -> float:
    # julichbrain leaves a region blank when it falls outside a subject's warp
    return float(cell) if cell else float("nan")


def _block(subjects, tivs, rows_of, columns, density=False):
    """One feature block: tissue volume / TIV per region, or GM density."""
    matrix, names = [], []
    for subject, tiv in zip(subjects, tivs):
        values, names = [], []
        for row in rows_of(subject):
            if density:
                region = _value(row["region_mm3"])
                values.append(_value(row["gmv_mm3"]) / region if region > 0 else 0.0)
                names.append(f"dens::{row['ROIname']}")
                continue
            for column in columns:
                values.append(_value(row[column]) / tiv)
                names.append(f"{column[:2]}::{row['ROIname']}")
        matrix.append(values)
    matrix = np.asarray(matrix, dtype=np.float64)
    keep = matrix.std(axis=0) > 0 if density else np.nanmedian(matrix, axis=0) > 1e-5
    return matrix[:, keep], [n for n, k in zip(names, keep) if k]


def extended_features(subjects, dmp_dir: str, multi_dir: str):
    """Neuromorphometrics GM/WM/CSF volumes and GM density, plus GM volume in each
    of ``EXTRA_ATLASES``; columns that are empty or missing for anyone are dropped."""
    def tiv(subject):
        with open(os.path.join(dmp_dir, f"tiv{subject}.csv"), encoding="utf-8") as handle:
            return float(list(csv.DictReader(handle))[0]["tiv_cm3"]) * 1000.0

    tivs = [tiv(s) for s in subjects]
    blocks = [
        _block(subjects, tivs, lambda s: _region_rows(dmp_dir, ATLAS, s),
               ("gmv_mm3", "wmv_mm3", "csfv_mm3")),
        _block(subjects, tivs, lambda s: _region_rows(dmp_dir, ATLAS, s), (), density=True),
    ] + [
        _block(subjects, tivs, lambda s, a=atlas: _region_rows(multi_dir, a, s), ("gmv_mm3",))
        for atlas in EXTRA_ATLASES
    ]
    prefixes = [ATLAS, ATLAS, *EXTRA_ATLASES]
    matrix = np.hstack([b[0] for b in blocks])
    names = [f"{p}:{n}" for p, b in zip(prefixes, blocks) for n in b[1]]
    complete = ~np.isnan(matrix).any(axis=0)
    return matrix[:, complete], [n for n, k in zip(names, complete) if k]


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="deepmriprep region morphometry")
    parser.add_argument("--data-dir", default=os.path.join("data", "mni2mm"))
    parser.add_argument("--out", default=os.path.join("data", "mni2mm", "features_vbm.npz"))
    parser.add_argument("--extended-out", default=None,
                        help="also build the multi-atlas set (reruns deepmriprep for EXTRA_ATLASES)")
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

    if args.extended_out:
        from deepmriprep import run_preprocess

        multi_dir = os.path.join(args.data_dir, "dmp_multi")
        subjects = [r["subject_id"] for r in kept]
        todo = [os.path.join(args.data_dir, "bet", f"{s}.nii.gz") for s in subjects
                if not os.path.exists(os.path.join(multi_dir, f"{EXTRA_ATLASES[-1]}_volumes{s}.csv"))]
        if todo:
            run_preprocess(input_paths=todo, output_dir=multi_dir, dir_format="flat",
                           outputs=[f"{a}_volumes" for a in EXTRA_ATLASES])
        extended, ext_names = extended_features(subjects, out_dir, multi_dir)
        save_features(args.extended_out, extended.astype(np.float32),
                      np.asarray([int(r["label"]) for r in kept], dtype=np.int64),
                      subjects, [r["site_id"] for r in kept], ext_names)
        print(f"wrote {args.extended_out} with shape {extended.shape[0]}x{extended.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
