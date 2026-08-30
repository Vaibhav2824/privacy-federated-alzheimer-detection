"""
orientation_search.py — the registration-scored half of the orientation search.

Kept apart from ``orientation`` because everything here needs a downloaded MNI
template and a real registration to run, so it is exercised by the pipeline runs
rather than by the unit suite, while the array logic it calls is pure and fully
tested.
"""

from __future__ import annotations

import numpy as np

from .orientation import apply_orientation, shape_cube, shortlist_by_shape


class CoarseTarget:
    """A 4 mm whole-head MNI target used to score candidate orientations.

    Scoring a candidate by *actually registering it* is what separates right
    from wrong; scale-and-position-free shape comparison leaves the top
    candidates within 0.05 of each other.  The target is the ICBM152 2009
    whole-head T1 rather than a skull-stripped brain, because the scan being
    oriented still has its skull: matching head to head scores the correct
    orientation around 0.74-0.79 with a clear step down to the third candidate,
    where matching an unstripped head against a stripped brain scored 0.09-0.48
    and picked a different, wrong orientation on the same subjects.  Working at
    4 mm keeps a candidate to about ten seconds.

    The one pair the score cannot separate is a left-right mirror, which stays
    within about 0.01.  That is resolved separately, against subjects that have
    both a degenerate-affine scan and one whose header is usable.
    """

    def __init__(self, template=None, template_mask=None, resolution: int = 4):
        import nibabel as nib
        from nilearn.datasets import fetch_icbm152_2009
        from nilearn.image import resample_img
        from scipy import ndimage as ndi

        from .preprocess_v3 import _scale_to_unit

        head = nib.load(fetch_icbm152_2009()["t1"])
        affine = head.affine.copy()
        affine[:3, :3] *= resolution
        image = resample_img(
            head, target_affine=affine, interpolation="continuous",
            force_resample=True, copy_header=True,
        )
        data = np.nan_to_num(np.asarray(image.dataobj, dtype=np.float32))
        data = _scale_to_unit(data, data > 0)

        self.image = image
        self.affine = image.affine
        self.mask = data > 0.02
        self.static = data
        centre_vox = np.asarray(ndi.center_of_mass(data > 0.1), dtype=np.float64)
        self.centre_world = self.affine[:3, :3] @ centre_vox + self.affine[:3, 3]
        self.reference_cube = shape_cube(data)


def coarse_score(volume: np.ndarray, zooms, target: CoarseTarget) -> float:
    """Register one candidate at 4 mm and return its template correlation."""
    import nibabel as nib
    from dipy.align import affine_registration
    from nilearn.image import resample_to_img

    from .preprocess_v3 import _recentred_affine, _scale_to_unit, qc_correlation

    affine = np.eye(4)
    affine[:3, :3] = np.diag([float(z) for z in zooms])
    affine = _recentred_affine(volume, affine, target.centre_world)
    initial = resample_to_img(
        nib.Nifti1Image(volume, affine), target.image,
        interpolation="continuous", copy_header=True, force_resample=True,
    )
    moving = np.nan_to_num(np.asarray(initial.dataobj, dtype=np.float32))
    moving = _scale_to_unit(moving, moving > 0)
    warped, _ = affine_registration(
        moving, target.static,
        moving_affine=initial.affine, static_affine=target.affine,
        nbins=32, metric="MI", sampling_proportion=25,
        pipeline=["translation", "rigid", "affine"],
        level_iters=[200, 80], sigmas=[2.0, 0.0], factors=[2, 1],
    )
    warped = np.asarray(warped, dtype=np.float32)
    return qc_correlation(warped, target.static, target.mask)



def search_orientation(volume: np.ndarray, zooms, target: CoarseTarget,
                       shortlist: int = 16):
    """Best (permutation, flips) for one volume, by coarse registration score."""
    results = []
    for _, permutation, flips in shortlist_by_shape(
        volume, target.reference_cube, shortlist
    ):
        candidate = apply_orientation(volume, permutation, flips)
        candidate_zooms = [zooms[axis] for axis in permutation]
        results.append((coarse_score(candidate, candidate_zooms, target),
                        permutation, flips))
    results.sort(key=lambda item: -item[0])
    return results


