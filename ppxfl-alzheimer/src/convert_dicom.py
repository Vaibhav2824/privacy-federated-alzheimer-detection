"""Convert extracted ADNI DICOM series into per-series NIfTI files, routed into
the existing AD-150/CN-150/MCI-150 raw-data layout by research-group class.

Idempotent: skips any series whose output .nii already exists, so it can be
safely re-run after an interruption.
"""
import json
import os
import sys

import nibabel as nib
import numpy as np
import pydicom

STAGING = sys.argv[1] if len(sys.argv) > 1 else "dicom_staging"
DATA_ROOT = sys.argv[2] if len(sys.argv) > 2 else "."
CLASS_MAP_PATH = sys.argv[3] if len(sys.argv) > 3 else "subject_class_map.json"

CLASS_DIR = {"AD": "AD-150", "CN": "CN-150", "MCI": "MCI-150"}


def convert_series(series_dir, out_path):
    dcm_files = [f for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    if not dcm_files:
        return False
    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, f))
            slices.append(ds)
        except Exception:
            continue
    if not slices:
        return False

    def sort_key(ds):
        if hasattr(ds, "InstanceNumber"):
            return int(ds.InstanceNumber)
        return 0

    slices.sort(key=sort_key)
    try:
        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=-1)
    except Exception:
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), out_path)
    return True


def main():
    class_map = json.load(open(CLASS_MAP_PATH))
    adni_root = os.path.join(STAGING, "ADNI")
    subjects = sorted(os.listdir(adni_root))

    log_path = os.path.join(STAGING, "convert_log.txt")
    done = set()
    if os.path.exists(log_path):
        done = set(open(log_path).read().split())

    converted = 0
    skipped = 0
    failed = []

    for subj in subjects:
        cls = class_map.get(subj)
        if cls is None:
            continue
        subj_dir = os.path.join(adni_root, subj)
        for seq in os.listdir(subj_dir):
            seq_dir = os.path.join(subj_dir, seq)
            for date in os.listdir(seq_dir):
                date_dir = os.path.join(seq_dir, date)
                for image_id in os.listdir(date_dir):
                    series_dir = os.path.join(date_dir, image_id)
                    key = f"{subj}/{seq}/{date}/{image_id}"
                    if key in done:
                        skipped += 1
                        continue
                    out_path = os.path.join(
                        DATA_ROOT, CLASS_DIR[cls], "ADNI", subj, seq, date, image_id,
                        f"{subj}_{seq}_{image_id}.nii",
                    )
                    if os.path.exists(out_path):
                        with open(log_path, "a") as lf:
                            lf.write(key + "\n")
                        skipped += 1
                        continue
                    ok = convert_series(series_dir, out_path)
                    with open(log_path, "a") as lf:
                        lf.write(key + "\n")
                    if ok:
                        converted += 1
                    else:
                        failed.append(key)
                    if (converted + skipped) % 25 == 0:
                        print(f"progress: converted={converted} skipped={skipped} failed={len(failed)}", flush=True)

    print(f"DONE converted={converted} skipped={skipped} failed={len(failed)}")
    if failed:
        print("FAILED:", failed[:20])


if __name__ == "__main__":
    main()
