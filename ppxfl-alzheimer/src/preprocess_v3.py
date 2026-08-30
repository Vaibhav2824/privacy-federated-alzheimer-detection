"""
preprocess_v3.py — anatomically standardised ADNI T1 preprocessing.

Replaces the v1/v2 pipeline, which selected a slice axis with
``np.argmax(volume.shape)``.  For an ADNI cohort that mixes RAS-stored
(256, 240, 160) volumes with IPL-stored (256, 256, 166) volumes, that axis is
the sagittal axis for some subjects and the axial axis for others, so the
"middle axial slice" was a different anatomical plane depending on how the
scanner happened to store the volume.  Nothing downstream can recover from
that, which is why the v2 cohort sat at chance.

This pipeline instead puts every subject in the same space:

1. one scan per subject, chosen deterministically by preprocessing level then
   by earliest acquisition date (ADNI ``Mask`` series are excluded: they are
   derived binary products, not intensity images);
2. skull stripping with deepbet;
3. reorientation to canonical RAS and resampling onto the MNI152 2 mm grid
   using the scan's own affine;
4. mutual-information affine registration (translation -> rigid -> affine) to
   the MNI152 template, so voxel (i, j, k) means the same anatomy in every
   subject;
5. intensity scaling inside the brain mask;
6. a per-subject registration QC score, so failures are excluded by a recorded
   rule rather than by eye.

The output is one ``float16`` volume per subject on the MNI 2 mm grid, plus a
manifest carrying subject, site, class and provenance.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

SUBJECT_ID_RE = re.compile(r"\d{3}_S_\d{4}")
SCAN_ID_RE = re.compile(r"I\d+")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_\d{2}_\d{2}_\d{2}")

CLASS_DIRS = {"CN": "CN-150", "MCI": "MCI-150", "AD": "AD-150"}
LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}

# Preferred ADNI preprocessing level, most processed first.  Higher rank wins.
SERIES_RANK = [
    "N3__Scaled_2",
    "N3__Scaled",
    "N3",
    "B1_Correction",
    "GradWarp",
]

# Series whose images are derived binary/masked products rather than intensity
# volumes.  Registering these produces a uniform blob.
EXCLUDED_SERIES_TOKENS = ("Mask", "HarP")

MNI_RESOLUTION = 2
QC_MIN_CORRELATION = 0.30


@dataclass
class ScanRecord:
    subject_id: str
    site_id: str
    klass: str
    label: int
    scan_id: str
    series: str
    acq_date: str
    source_path: str


def _series_of(path: str) -> str:
    """The ADNI series directory name, e.g. ``MPR-R__GradWarp__N3``."""
    parts = os.path.normpath(path).split(os.sep)
    for i, part in enumerate(parts):
        if SUBJECT_ID_RE.fullmatch(part) and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def _acq_date_of(path: str) -> str:
    match = DATE_RE.search(path.replace(os.sep, "/"))
    return match.group(1) if match else "9999-99-99"


def _series_rank(series: str) -> int:
    for rank, token in enumerate(SERIES_RANK):
        if token in series:
            return len(SERIES_RANK) - rank
    return 0


def discover_scans(data_root: str) -> list[ScanRecord]:
    """Every candidate NIfTI in the three class folders, as records."""
    records: list[ScanRecord] = []
    for klass, folder in CLASS_DIRS.items():
        pattern = os.path.join(data_root, folder, "**", "*.nii*")
        for path in sorted(glob.glob(pattern, recursive=True)):
            match = SUBJECT_ID_RE.search(path.replace(os.sep, "/"))
            if match is None:
                continue
            subject_id = match.group(0)
            series = _series_of(path)
            if any(token in series for token in EXCLUDED_SERIES_TOKENS):
                continue
            scan_match = SCAN_ID_RE.search(os.path.basename(path))
            records.append(
                ScanRecord(
                    subject_id=subject_id,
                    site_id=subject_id.split("_")[0],
                    klass=klass,
                    label=LABEL_MAP[klass],
                    scan_id=scan_match.group(0) if scan_match else os.path.basename(path),
                    series=series,
                    acq_date=_acq_date_of(path),
                    source_path=path,
                )
            )
    return records


def select_one_per_subject(records: list[ScanRecord]) -> list[ScanRecord]:
    """One scan per subject: most processed series, then earliest visit.

    Multiple ADNI preprocessing levels of the same acquisition are present in
    the download (GradWarp, B1_Correction, N3, Scaled).  Treating them as
    separate samples inflates the dataset with near-duplicates of the same
    brain, so exactly one is kept per subject.
    """
    best: dict[str, ScanRecord] = {}
    for record in records:
        key = record.subject_id
        current = best.get(key)
        if current is None:
            best[key] = record
            continue
        candidate_sort = (-_series_rank(record.series), record.acq_date, record.scan_id)
        current_sort = (-_series_rank(current.series), current.acq_date, current.scan_id)
        if candidate_sort < current_sort:
            best[key] = record
    return [best[k] for k in sorted(best)]


def _templates():
    from nilearn.datasets import load_mni152_brain_mask, load_mni152_template

    template = load_mni152_template(resolution=MNI_RESOLUTION)
    mask = np.asarray(load_mni152_brain_mask(resolution=MNI_RESOLUTION).dataobj) > 0
    return template, mask


def _scale_to_unit(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = volume[mask]
    if values.size == 0:
        return volume.astype(np.float32)
    high = float(np.percentile(values, 99.5))
    if high <= 0:
        return np.zeros_like(volume, dtype=np.float32)
    return (np.clip(volume, 0.0, high) / high).astype(np.float32) * mask


def _recentred_affine(data: np.ndarray, affine: np.ndarray, target_world: np.ndarray):
    """``affine`` with its translation set so the brain centroid is at the target.

    ADNI scan affines encode scanner coordinates, whose origin sits wherever the
    scanner's isocentre happened to be.  Resampling straight onto the MNI grid
    with those affines drops many subjects outside the template's field of view,
    leaving a nearly empty volume that mutual information cannot recover from.
    Only the translation is replaced: the rotation block carries the oblique and
    head-tilt correction that the acquisition actually recorded, and discarding
    it leaves obliquely acquired scans 90 degrees out and unrecoverable by the
    subsequent affine search.
    """
    from scipy import ndimage as ndi

    centre_vox = np.asarray(ndi.center_of_mass(data > 0), dtype=np.float64)
    out = np.asarray(affine, dtype=np.float64).copy()
    out[:3, 3] = np.asarray(target_world) - out[:3, :3] @ centre_vox
    return out


# Registration schedules, tried in order until one clears the QC threshold.
_SCHEDULES = (
    {
        "pipeline": ["translation", "rigid", "affine"],
        "level_iters": [1000, 200, 50],
        "sigmas": [3.0, 1.0, 0.0],
        "factors": [4, 2, 1],
    },
    {
        "pipeline": ["center_of_mass", "translation", "rigid", "affine"],
        "level_iters": [2000, 1000, 400, 100],
        "sigmas": [5.0, 3.0, 1.0, 0.0],
        "factors": [8, 4, 2, 1],
    },
)


def orient_raw_scan(path: str, orientation_table: dict, target=None):
    """Load a raw scan in a defensible anatomical orientation.

    A scan with a usable affine is simply reoriented to canonical RAS.  A scan
    whose affine is the identity carries no orientation at all, so the fitted
    table for its (series family, array shape) group supplies one; if the group
    is unknown, the orientation is searched for this scan directly.

    Returns ``(data, affine, source)`` where ``source`` records which of the
    three routes was taken, so the manifest can report it.
    """
    import nibabel as nib

    from .orientation import apply_orientation, group_key
    from .orientation_search import search_orientation

    image = nib.load(path)
    zooms = image.header.get_zooms()[:3]
    if not np.allclose(image.affine, np.eye(4)):
        canonical = nib.as_closest_canonical(image)
        data = np.nan_to_num(np.asarray(canonical.dataobj, dtype=np.float32))
        if data.ndim == 4:
            data = data[..., 0]
        return data, canonical.affine, "header"

    data = np.nan_to_num(np.asarray(image.dataobj, dtype=np.float32))
    if data.ndim == 4:
        data = data[..., 0]

    key = group_key(_series_of(path), data.shape)
    entry = orientation_table.get(key)
    if entry is not None:
        permutation, flips = entry
        source = "table"
    elif target is not None:
        _, permutation, flips = search_orientation(data, zooms, target)[0]
        source = "searched"
    else:
        return data, np.diag([*[float(z) for z in zooms], 1.0]), "unoriented"

    oriented = apply_orientation(data, permutation, flips)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([float(zooms[axis]) for axis in permutation])
    return oriented, affine, source


def register_volume_to_mni(data: np.ndarray, affine: np.ndarray, template,
                           template_mask: np.ndarray, static_scaled: np.ndarray):
    """Affine-register one skull-stripped brain onto the MNI 2 mm grid.

    Returns ``(volume, qc_correlation)``.  Each schedule in ``_SCHEDULES`` is
    tried until one clears ``QC_MIN_CORRELATION``; the best result is returned
    either way so that failures are recorded rather than silently dropped.
    """
    import nibabel as nib
    from dipy.align import affine_registration
    from nilearn.image import resample_to_img
    from scipy import ndimage as ndi

    shape = np.asarray(template.dataobj).shape
    if not (data > 0).any():
        return np.zeros(shape, dtype=np.float32), 0.0

    centre_vox = np.asarray(ndi.center_of_mass(template_mask), dtype=np.float64)
    centre_world = template.affine[:3, :3] @ centre_vox + template.affine[:3, 3]
    recentred = nib.Nifti1Image(data, _recentred_affine(data, affine, centre_world))

    initial = resample_to_img(
        recentred, template, interpolation="continuous", copy_header=True, force_resample=True
    )
    moving = np.nan_to_num(np.asarray(initial.dataobj, dtype=np.float32))
    moving = _scale_to_unit(moving, moving > 0)

    best_volume = np.zeros(shape, dtype=np.float32)
    best_score = -1.0
    for schedule in _SCHEDULES:
        warped, _ = affine_registration(
            moving,
            static_scaled,
            moving_affine=initial.affine,
            static_affine=template.affine,
            nbins=32,
            metric="MI",
            **schedule,
        )
        volume = _scale_to_unit(
            np.asarray(warped, dtype=np.float32) * template_mask, template_mask
        )
        score = qc_correlation(volume, static_scaled, template_mask)
        if score > best_score:
            best_volume, best_score = volume, score
        if score >= QC_MIN_CORRELATION:
            break
    return best_volume, best_score


def qc_correlation(volume: np.ndarray, template: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation with the template inside the brain mask."""
    a = volume[mask].astype(np.float64)
    b = template[mask].astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def write_manifest(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _orient_and_strip(records, out_dir: str, orientation_table: dict, target,
                      batch_size: int = 24) -> None:
    """Write each scan in its measured orientation and skull-strip it.

    Orientation has to come first: deepbet is trained on anatomically oriented
    volumes and leaves face and neck behind when given a scan lying on its side,
    which then defeats registration too.

    Work proceeds in batches, and each batch's oriented volumes are deleted as
    soon as it has been stripped.  Orienting the whole cohort up front means
    holding an uncompressed copy of every scan at once -- for this cohort about
    25 GB -- which exhausts both the disk and the memory the reader maps, and
    deepbet then fails the entire run rather than one batch of it.
    """
    import nibabel as nib
    from deepbet import run_bet

    oriented_dir = os.path.join(out_dir, "oriented")
    bet_dir = os.path.join(out_dir, "bet")
    os.makedirs(oriented_dir, exist_ok=True)
    os.makedirs(bet_dir, exist_ok=True)

    sources_path = os.path.join(out_dir, "orientation_sources.json")
    sources: dict[str, str] = {}
    if os.path.exists(sources_path):
        with open(sources_path, encoding="utf-8") as handle:
            sources = json.load(handle)

    todo = [r for r in records
            if not os.path.exists(os.path.join(bet_dir, f"{r.subject_id}.nii.gz"))]
    if not todo:
        return
    print(f"skull stripping {len(todo)} volumes in batches of {batch_size}",
          flush=True)

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        pending, brains = [], []
        for record in batch:
            try:
                data, affine, source = orient_raw_scan(
                    record.source_path, orientation_table, target
                )
            except Exception as error:  # a single unreadable scan must not stop the run
                print(f"  [skip] {record.subject_id}: {error}", flush=True)
                continue
            sources[record.subject_id] = source
            oriented_path = os.path.join(oriented_dir, f"{record.subject_id}.nii.gz")
            nib.save(nib.Nifti1Image(data.astype(np.float32), affine), oriented_path)
            pending.append(oriented_path)
            brains.append(os.path.join(bet_dir, f"{record.subject_id}.nii.gz"))
            del data

        if pending:
            try:
                run_bet(pending, brain_paths=brains, n_dilate=0)
            except Exception as error:
                print(f"  [batch failed] {error}", flush=True)
            for path in pending:
                if os.path.exists(path):
                    os.remove(path)

        done = min(start + batch_size, len(todo))
        print(f"  stripped {done}/{len(todo)}", flush=True)
        with open(sources_path, "w", encoding="utf-8") as handle:
            json.dump(sources, handle, indent=2)

    counts = collections.Counter(sources.values())
    print(f"orientation sources: {dict(counts)}", flush=True)


def run(data_root: str, out_dir: str, limit: int | None, shard: int, num_shards: int,
        orientation_path: str, stage: str = "all") -> None:
    import nibabel as nib

    from .orientation import load_table
    from .orientation_search import CoarseTarget

    template, template_mask = _templates()
    template_data = _scale_to_unit(
        np.asarray(template.dataobj, dtype=np.float32) * template_mask, template_mask
    )
    target = CoarseTarget(template, template_mask)
    orientation_table = load_table(orientation_path)
    print(f"orientation table: {len(orientation_table)} groups from {orientation_path}")

    records = select_one_per_subject(discover_scans(data_root))
    if limit is not None:
        records = records[:limit]
    print(f"selected {len(records)} subjects "
          f"({len({r.site_id for r in records})} ADNI sites)")

    bet_dir = os.path.join(out_dir, "bet")
    vol_dir = os.path.join(out_dir, "vol")
    os.makedirs(vol_dir, exist_ok=True)

    # Skull stripping must finish for every subject before any shard starts
    # registering, otherwise a shard reaches a subject whose stripped brain does
    # not exist yet and skips it.  It runs on the GPU and is cheap, so it is a
    # separate stage rather than something shard 0 races the others to finish.
    if stage in ("bet", "all"):
        _orient_and_strip(records, out_dir, orientation_table, target)
    if stage == "bet":
        print("skull stripping complete; run --stage register next")
        return

    rows = []
    started = time.time()
    if num_shards > 1:
        records = [r for i, r in enumerate(records) if i % num_shards == shard]
        print(f"shard {shard}/{num_shards}: {len(records)} subjects")
    for index, record in enumerate(records, start=1):
        brain_path = os.path.join(bet_dir, f"{record.subject_id}.nii.gz")
        vol_path = os.path.join(vol_dir, f"{record.subject_id}.npy")
        if os.path.exists(vol_path):
            volume = np.load(vol_path).astype(np.float32)
            score = qc_correlation(volume, template_data, template_mask)
        else:
            if not os.path.exists(brain_path):
                print(f"  [skip] no stripped brain for {record.subject_id}")
                continue
            brain = nib.load(brain_path)
            data = np.nan_to_num(np.asarray(brain.dataobj, dtype=np.float32))
            volume, score = register_volume_to_mni(
                data, brain.affine, template, template_mask, template_data
            )
            np.save(vol_path, volume.astype(np.float16))
        row = asdict(record)
        row["qc_correlation"] = round(score, 4)
        row["qc_pass"] = int(score >= QC_MIN_CORRELATION)
        row["volume_path"] = os.path.relpath(vol_path, out_dir)
        rows.append(row)
        elapsed = time.time() - started
        print(f"  [{index}/{len(records)}] {record.subject_id} {record.klass} "
              f"r={score:.3f} ({elapsed / index:.1f}s/subject)", flush=True)

    if num_shards > 1:
        print(f"shard {shard} done; rerun with --num-shards 1 to write the manifest")
        return

    write_manifest(os.path.join(out_dir, "manifest.csv"), rows)
    passed = sum(r["qc_pass"] for r in rows)
    summary = {
        "subjects": len(rows),
        "qc_passed": passed,
        "qc_failed": len(rows) - passed,
        "qc_threshold": QC_MIN_CORRELATION,
        "sites": len({r["site_id"] for r in rows}),
        "grid": "MNI152 2mm",
        "shape": list(np.asarray(template.dataobj).shape),
        "class_counts": {
            k: sum(1 for r in rows if r["klass"] == k and r["qc_pass"])
            for k in CLASS_DIRS
        },
    }
    with open(os.path.join(out_dir, "preprocess_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ADNI T1 -> MNI152 2mm preprocessing")
    parser.add_argument("--data-root", default="..",
                        help="directory holding AD-150 / MCI-150 / CN-150")
    parser.add_argument("--out", default=os.path.join("data", "mni2mm"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0,
                        help="index of this shard when registering in parallel")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--stage", choices=["all", "bet", "register"],
                        default="all",
                        help="bet strips every subject; register is shardable")
    parser.add_argument("--orientation-table",
                        default=os.path.join("data", "orientation_table.json"))
    args = parser.parse_args(argv)
    run(args.data_root, args.out, args.limit, args.shard, args.num_shards,
        args.orientation_table, args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
