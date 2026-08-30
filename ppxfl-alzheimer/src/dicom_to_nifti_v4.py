"""
dicom_to_nifti_v4.py — convert ADNI DICOM series to NIfTI, keeping the geometry.

The converter used for the earlier expansion stacked slices by InstanceNumber
and saved them with ``affine=np.eye(4)``.  That discards every piece of spatial
information the scanner recorded: voxel sizes, slice order, patient
orientation.  It is the direct cause of the 272 identity-affine scans in the
current cohort, and therefore of the 48-candidate orientation search that
``orientation.py`` exists to perform.  Converting correctly in the first place
removes that whole problem for newly ingested subjects.

Geometry is delegated to SimpleITK's GDCM series reader rather than
reconstructed by hand.  It sorts slices by position along the slice normal
instead of by instance number (which is not always monotonic in slice order),
applies the rescale slope and intercept, and derives the direction cosines,
origin and spacing from the series, so the written NIfTI carries a real affine
that ``nibabel.as_closest_canonical`` can act on.

Idempotent: a series whose output already exists is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

SUBJECT_RE = re.compile(r"\d{3}_S_\d{4,5}")
CLASS_DIRS = {"CN": "CN-150", "MCI": "MCI-150", "AD": "AD-150"}

# Series that are acquired alongside the structural scan but are not it.
NON_STRUCTURAL = re.compile(
    r"(localizer|scout|calibration|field ?map|survey|rsfMRI|fMRI|DTI|ASL|"
    r"perfusion|B1|smartbrain)",
    re.IGNORECASE,
)


def labels_from_export(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            subject, group = row.get("Subject ID"), row.get("Research Group")
            if subject and group in CLASS_DIRS:
                out.setdefault(subject, group)
    return out


def find_series(staging_root: str):
    """Every leaf directory holding DICOM files, with its ADNI path parts."""
    for root, _dirs, files in os.walk(staging_root):
        if not any(f.lower().endswith(".dcm") for f in files):
            continue
        parts = root.replace("\\", "/").split("/")
        match = SUBJECT_RE.search(root.replace("\\", "/"))
        if match is None:
            continue
        subject = match.group(0)
        index = parts.index(subject) if subject in parts else -1
        series = parts[index + 1] if 0 <= index < len(parts) - 1 else "UNKNOWN"
        image_id = parts[-1]
        date = parts[index + 2] if 0 <= index < len(parts) - 2 else "UNKNOWN"
        yield subject, series, date, image_id, root


def convert_series(series_dir: str, out_path: str) -> bool:
    """Write one DICOM series to NIfTI with its recorded geometry."""
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(series_dir)
    if not names:
        return False
    reader.SetFileNames(names)
    try:
        image = reader.Execute()
    except Exception:
        return False
    if image.GetDimension() != 3 or min(image.GetSize()) < 2:
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(image, out_path)
    return True


def run(staging: str, data_root: str, export: str, skip_non_structural: bool,
        prune: bool = False, min_free_gb: float = 3.0) -> dict:
    """Convert every series under ``staging``.

    With ``prune``, a series' DICOM files are deleted as soon as it has been
    converted.  The archives and their converted output do not fit on disk at
    the same time -- roughly 10 GB of DICOM expands to roughly 21 GB of NIfTI --
    so holding the whole staging tree until the end runs the disk to zero and
    SimpleITK begins failing writes.  Freeing each series as it is consumed
    keeps peak usage near the larger of the two rather than their sum.
    """
    import shutil

    labels = labels_from_export(export)
    stats = {"converted": 0, "skipped": 0, "unlabelled": 0, "failed": 0,
             "non_structural": 0, "pruned": 0}
    subjects: set[str] = set()

    def free_gb() -> float:
        return shutil.disk_usage(data_root).free / (1024 ** 3)

    for subject, series, date, image_id, path in find_series(staging):
        group = labels.get(subject)
        if group is None:
            stats["unlabelled"] += 1
            continue
        if skip_non_structural and NON_STRUCTURAL.search(series):
            stats["non_structural"] += 1
            continue
        out_path = os.path.join(
            data_root, CLASS_DIRS[group], "ADNI", subject, series, date, image_id,
            f"{subject}_{series}_{image_id}.nii",
        )
        if os.path.exists(out_path):
            stats["skipped"] += 1
            subjects.add(subject)
            if prune:
                shutil.rmtree(path, ignore_errors=True)
                stats["pruned"] += 1
            continue

        if free_gb() < min_free_gb:
            print(f"  [stop] only {free_gb():.1f} GB free, below the "
                  f"{min_free_gb} GB floor; re-run after making room", flush=True)
            break

        if convert_series(path, out_path):
            stats["converted"] += 1
            subjects.add(subject)
            if prune:
                shutil.rmtree(path, ignore_errors=True)
                stats["pruned"] += 1
        else:
            stats["failed"] += 1
        total = stats["converted"] + stats["skipped"]
        if total and total % 25 == 0:
            print(f"  {total} series done ({len(subjects)} subjects), "
                  f"{free_gb():.1f} GB free", flush=True)

    stats["subjects"] = len(subjects)
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ADNI DICOM to NIfTI")
    parser.add_argument("--staging", required=True,
                        help="directory holding the extracted ADNI/ tree")
    parser.add_argument("--data-root", default="..")
    parser.add_argument("--export", default=os.path.join("data", "ida_search_v4.csv"))
    parser.add_argument("--keep-non-structural", action="store_true")
    parser.add_argument("--prune-staging", action="store_true",
                        help="delete each series' DICOM once it is converted")
    parser.add_argument("--min-free-gb", type=float, default=3.0,
                        help="stop rather than fill the disk below this")
    args = parser.parse_args(argv)

    stats = run(args.staging, args.data_root, args.export,
                not args.keep_non_structural, args.prune_staging,
                args.min_free_gb)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
