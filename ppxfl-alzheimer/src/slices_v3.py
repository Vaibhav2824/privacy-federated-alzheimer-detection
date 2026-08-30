"""
slices_v3.py — fixed-coordinate 2.5D slices from MNI-registered volumes.

The previous pipeline took its slices at ``n // 2`` of whichever array axis
happened to be longest, so the plane and the anatomy both varied by subject.
Here slices are taken at named MNI millimetre coordinates, which means slice
*k* shows the same structures in every subject and a model can learn a spatial
pattern rather than an average over unrelated views.

Levels are chosen where Alzheimer's structural change is best established:
axial levels from the medial temporal lobe up through the lateral ventricles,
and coronal levels through the hippocampal head and body.  Each sample stacks
three adjacent parallel slices as three channels, which lets an ImageNet
backbone be used as trained instead of having its first convolution collapsed
to a single channel.
"""

from __future__ import annotations

import numpy as np

# MNI millimetre coordinates of the extracted levels.
AXIAL_MM = (-30, -24, -18, -12, -6, 0, 6, 12, 18, 24)
CORONAL_MM = (-40, -32, -24, -16, -8)
CHANNEL_OFFSET_MM = 4


def mm_to_voxel(affine: np.ndarray, mm) -> np.ndarray:
    inverse = np.linalg.inv(affine)
    return inverse[:3, :3] @ np.asarray(mm, dtype=float) + inverse[:3, 3]


def _take(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    index = int(np.clip(index, 0, volume.shape[axis] - 1))
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def _resize(plane: np.ndarray, size: int) -> np.ndarray:
    from scipy import ndimage as ndi

    if plane.shape == (size, size):
        return plane.astype(np.float32)
    zoom = (size / plane.shape[0], size / plane.shape[1])
    out = ndi.zoom(plane.astype(np.float32), zoom, order=1)
    out = out[:size, :size]
    return np.pad(out, [(0, size - out.shape[0]), (0, size - out.shape[1])])


def extract_slices(volume: np.ndarray, affine: np.ndarray, size: int = 224):
    """Three-channel slices at every configured MNI level.

    Returns ``(stack, descriptions)`` where ``stack`` is
    ``(n_levels, 3, size, size)`` and each description names the plane and the
    millimetre coordinate it was taken at.
    """
    voxel_per_mm = 1.0 / abs(float(affine[2, 2]))
    offset = max(1, int(round(CHANNEL_OFFSET_MM * voxel_per_mm)))

    planes, descriptions = [], []
    for axis, coordinates, letter in ((2, AXIAL_MM, "z"), (1, CORONAL_MM, "y")):
        for millimetre in coordinates:
            centre = mm_to_voxel(affine, (0.0, 0.0, 0.0))
            point = list(centre)
            point[axis] = mm_to_voxel(
                affine,
                tuple(millimetre if a == axis else 0.0 for a in range(3)),
            )[axis]
            index = int(round(point[axis]))
            channels = [
                _resize(_take(volume, axis, index + shift), size)
                for shift in (-offset, 0, offset)
            ]
            planes.append(np.stack(channels, axis=0))
            descriptions.append(f"{letter}={millimetre}")
    return np.stack(planes).astype(np.float32), descriptions


def build_slice_dataset(manifest_rows, volume_dir: str, affine: np.ndarray,
                        size: int = 224, progress: bool = True):
    """Slice tensors for every subject, kept with their subject and site labels."""
    import os

    stacks, labels, subjects, sites = [], [], [], []
    descriptions = None
    for index, row in enumerate(manifest_rows, start=1):
        volume = np.load(
            os.path.join(volume_dir, f"{row['subject_id']}.npy")
        ).astype(np.float32)
        stack, descriptions = extract_slices(volume, affine, size=size)
        stacks.append(stack.astype(np.float16))
        labels.append(int(row["label"]))
        subjects.append(row["subject_id"])
        sites.append(row["site_id"])
        if progress and index % 25 == 0:
            print(f"  slices {index}/{len(manifest_rows)}", flush=True)
    return (
        np.stack(stacks),
        np.asarray(labels, dtype=np.int64),
        subjects,
        sites,
        descriptions,
    )


def main(argv=None) -> int:
    import argparse
    import csv
    import os

    from .preprocess_v3 import _templates

    parser = argparse.ArgumentParser(description="MNI-standardised 2.5D slices")
    parser.add_argument("--data-dir", default=os.path.join("data", "mni2mm"))
    parser.add_argument("--out", default=os.path.join("data", "mni2mm", "slices.npz"))
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args(argv)

    template, _ = _templates()
    with open(os.path.join(args.data_dir, "manifest.csv"), encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["qc_pass"] == "1"]
    print(f"extracting slices for {len(rows)} subjects")

    stacks, labels, subjects, sites, descriptions = build_slice_dataset(
        rows, os.path.join(args.data_dir, "vol"), template.affine, size=args.size
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, X=stacks, y=labels,
        subjects=np.asarray(subjects), sites=np.asarray(sites),
        levels=np.asarray(descriptions),
    )
    print(f"wrote {args.out} with shape {stacks.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
