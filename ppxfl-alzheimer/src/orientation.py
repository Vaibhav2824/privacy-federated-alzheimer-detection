"""
orientation.py — recover the anatomical orientation of ADNI scans whose NIfTI
affine is degenerate.

272 of the 620 usable scans in this download (the raw ``MPRAGE`` and
``Accelerated_SAG_IR-SPGR`` series) carry an identity affine with
``sform_code = 2``.  ``nibabel.as_closest_canonical`` therefore reports them as
RAS and leaves the array order untouched, even though the acquisition is
sagittal and the stored axis order is not RAS at all.  Downstream that means
the same voxel index refers to different anatomy in different subjects, which
no amount of training can undo.

Rather than assume a conversion convention, the orientation is *measured*: each
of the 48 axis permutations and reflections is scored by how well it registers
to the MNI152 template, cheaply at coarse resolution, and the best is kept.
The search is fitted once per (series family, array shape) group — the
DICOM-to-NIfTI convention is a property of the conversion, not of the subject —
so the whole cohort shares one orientation per group and cannot drift
subject-to-subject.
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np

# All 48 axis permutations x reflections.  Mirror candidates are kept because a
# DICOM-to-NIfTI conversion can flip left/right, and excluding them would bake
# in the assumption this module exists to test.
PERMUTATIONS = tuple(itertools.permutations(range(3)))
FLIPS = tuple(itertools.product([False, True], repeat=3))


def apply_orientation(volume: np.ndarray, permutation, flips) -> np.ndarray:
    """Reorder and reflect ``volume`` by one candidate orientation."""
    out = np.transpose(volume, permutation)
    for axis, flip in enumerate(flips):
        if flip:
            out = np.flip(out, axis)
    return np.ascontiguousarray(out)


def downsample(volume: np.ndarray, factor: int = 3) -> np.ndarray:
    """Block-mean downsample, used to keep the 48-candidate sweep cheap.

    Orientation is a property of gross anatomy, so the sweep does not need full
    resolution; at 1 mm a 256^3 volume makes the sweep the dominant cost.
    """
    if factor <= 1:
        return volume
    trimmed = volume[
        : volume.shape[0] // factor * factor,
        : volume.shape[1] // factor * factor,
        : volume.shape[2] // factor * factor,
    ]
    if 0 in trimmed.shape:
        return volume
    return trimmed.reshape(
        trimmed.shape[0] // factor, factor,
        trimmed.shape[1] // factor, factor,
        trimmed.shape[2] // factor, factor,
    ).mean(axis=(1, 3, 5))


def shape_cube(volume: np.ndarray, size: int = 48) -> np.ndarray:
    """The head, cropped to its bounding box and resampled into a fixed cube.

    Scale and position are removed, so the only thing left to compare is gross
    anatomical shape.  Used as a cheap pre-filter before the expensive
    registration-based scoring.
    """
    from scipy import ndimage as ndi

    volume = np.nan_to_num(np.asarray(volume, dtype=np.float32))
    positive = volume[volume > 0]
    if positive.size == 0:
        return np.zeros((size,) * 3, dtype=np.float32)
    # The threshold is a quarter of the 60th percentile of the positive voxels,
    # so at least those voxels survive it and the bounding box is never empty.
    index = np.argwhere(volume > (np.percentile(positive, 60) * 0.25))
    low, high = index.min(0), index.max(0) + 1
    crop = volume[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    cube = ndi.zoom(crop, [size / s for s in crop.shape], order=1)[:size, :size, :size]
    cube = np.pad(cube, [(0, size - s) for s in cube.shape])
    high_value = np.percentile(cube[cube > 0], 99)
    return np.clip(cube, 0.0, high_value) / (high_value + 1e-8)


def shortlist_by_shape(volume: np.ndarray, reference_cube: np.ndarray, size: int):
    """Rank all 48 orientations by shape alone, cheaply, and keep the best."""
    small = downsample(volume, factor=3)
    ranked = []
    for permutation in PERMUTATIONS:
        for flips in FLIPS:
            cube = shape_cube(apply_orientation(small, permutation, flips))
            if cube.std() == 0:
                continue
            score = float(np.corrcoef(cube.ravel(), reference_cube.ravel())[0, 1])
            ranked.append((score, permutation, flips))
    ranked.sort(key=lambda item: -item[0])
    return ranked[:size]


def load_table(path: str) -> dict:
    """Load a fitted orientation table keyed by ``family|shape``."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        key: (tuple(value["permutation"]), tuple(bool(f) for f in value["flips"]))
        for key, value in raw.items()
    }


def group_key(series: str, shape) -> str:
    upper = series.upper()
    if "IR-SPGR" in upper:
        family = "IR-SPGR"
    elif "MPRAGE" in upper:
        family = "MPRAGE"
    else:
        family = upper
    return f"{family}|{'x'.join(str(s) for s in shape[:3])}"
