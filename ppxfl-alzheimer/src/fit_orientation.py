"""
fit_orientation.py — measure the array orientation of each degenerate-affine
ADNI series group and write it to a table.

The orientation of a scan whose NIfTI affine is the identity cannot be read
from its header, so it is measured instead.  Candidates are ranked cheaply by
gross head shape, and the shortlist is then scored by actually registering each
one to a 4 mm MNI target: on this cohort the correct orientation scores about
0.48 against about 0.15 for the runner-up, a margin that shape comparison alone
does not produce.

The fit is per (series family, array shape) group, not per subject.  The
DICOM-to-NIfTI convention belongs to the conversion, so a group answer both
aggregates votes over several subjects and cannot leave two subjects in the
same group in different orientations.  Subjects that nevertheless fail
registration QC under the group orientation are re-searched individually by
``preprocess_v3``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

from .orientation import group_key
from .orientation_search import CoarseTarget, search_orientation
from .preprocess_v3 import _templates, discover_scans


def _is_degenerate(path: str) -> bool:
    import nibabel as nib

    return bool(np.allclose(nib.load(path).affine, np.eye(4)))


def load_volume(path: str):
    import nibabel as nib

    image = nib.load(path)
    data = np.nan_to_num(np.asarray(image.dataobj, dtype=np.float32))
    if data.ndim == 4:
        data = data[..., 0]
    return data, image.header.get_zooms()[:3]


def orientation_determinant(permutation, flips) -> int:
    """Sign of the linear map this candidate applies to the voxel frame."""
    matrix = np.zeros((3, 3))
    for axis, source in enumerate(permutation):
        matrix[axis, source] = -1.0 if flips[axis] else 1.0
    return int(round(float(np.linalg.det(matrix))))


def group_degenerate_scans(data_root: str):
    import nibabel as nib

    groups: dict[str, list] = collections.defaultdict(list)
    for record in discover_scans(data_root):
        if not _is_degenerate(record.source_path):
            continue
        shape = nib.load(record.source_path).shape[:3]
        groups[group_key(record.series, shape)].append(record)
    return groups


def fit(data_root: str, out_path: str, per_group: int, shortlist: int,
        only_groups=None) -> dict:
    template, template_mask = _templates()
    target = CoarseTarget(template, template_mask)
    groups = group_degenerate_scans(data_root)

    table = {}
    for key in sorted(groups, key=lambda k: -len(groups[k])):
        if only_groups and key not in only_groups:
            continue
        members = groups[key]
        seen, sample = set(), []
        for record in members:
            if record.subject_id in seen:
                continue
            seen.add(record.subject_id)
            sample.append(record)
            if len(sample) >= per_group:
                break

        votes = collections.Counter()
        totals = collections.defaultdict(float)
        best_scores = collections.defaultdict(list)
        margins = []
        for record in sample:
            volume, zooms = load_volume(record.source_path)
            results = search_orientation(volume, zooms, target, shortlist=shortlist)
            if not results:
                continue
            # Every candidate contributes its registration score, not just the
            # per-subject winner.  A subject whose search failed outright has
            # low scores everywhere and therefore little influence, whereas a
            # plurality vote would let it count as much as a clean one.
            for score, permutation, flips in results:
                totals[(permutation, flips)] += max(score, 0.0)
            score, permutation, flips = results[0]
            votes[(permutation, flips)] += 1
            best_scores[(permutation, flips)].append(score)
            if len(results) > 1:
                margins.append(score - results[1][0])
            print(f"  {key} {record.subject_id} -> perm={permutation} "
                  f"flips={flips} score={score:.3f} "
                  f"margin={score - results[1][0]:.3f}", flush=True)

        if not totals:
            continue
        permutation, flips = max(totals, key=totals.get)
        count = votes[(permutation, flips)]
        table[key] = {
            "permutation": list(permutation),
            "flips": [bool(f) for f in flips],
            "determinant": orientation_determinant(permutation, flips),
            "votes": count,
            "sampled": len(sample),
            "scans_in_group": len(members),
            "mean_score": round(float(np.mean(best_scores[(permutation, flips)])), 4)
            if best_scores[(permutation, flips)] else None,
            "total_score": round(float(totals[(permutation, flips)]), 4),
            "mean_margin": round(float(np.mean(margins)), 4) if margins else None,
            "alternatives": [
                {"permutation": list(p), "flips": [bool(x) for x in f], "votes": v}
                for (p, f), v in votes.most_common() if (p, f) != (permutation, flips)
            ],
        }
        print(f"{key}: perm={permutation} flips={flips} "
              f"({count}/{len(sample)} votes)", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2)
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="fit ADNI array orientations")
    parser.add_argument("--data-root", default="..")
    parser.add_argument("--out", default=os.path.join("data", "orientation_table.json"))
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--groups", nargs="*", default=None,
                        help="restrict the fit to these group keys")
    parser.add_argument("--shortlist", type=int, default=20)
    args = parser.parse_args(argv)
    table = fit(args.data_root, args.out, args.per_group, args.shortlist,
                only_groups=args.groups)
    print(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
