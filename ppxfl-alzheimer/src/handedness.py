"""
handedness.py — resolve the left-right ambiguity the orientation search leaves.

Registration score cannot separate an orientation from its left-right mirror:
the brain is close enough to symmetric that the two candidates land within
about 0.01 of each other.  Every other axis is settled by the search; this one
is not, and getting it wrong would mirror the whole degenerate-affine group
relative to the rest of the cohort.

It is settled against anchor subjects instead.  Four subjects in this download
have both a scan whose header is usable — whose handedness is therefore known —
and a scan from a degenerate-affine group.  Registering the same subject's two
scans and comparing the mirror candidates uses that subject's own asymmetry,
which is far stronger than the population-average asymmetry a template
comparison would have to rely on.

If no anchor exists for a group, the ambiguity is reported rather than guessed,
and the paper avoids lateralised claims for that group.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

from .orientation import apply_orientation, group_key, load_table
from .preprocess_v3 import _scale_to_unit, _templates, discover_scans, qc_correlation, register_volume_to_mni


def mirror(flips) -> tuple:
    """The same orientation with left and right exchanged.

    After the permutation is applied the array is treated as RAS-ordered, so
    output axis 0 is the right-left axis and toggling its flip is exactly the
    left-right mirror.
    """
    flipped = list(flips)
    flipped[0] = not flipped[0]
    return tuple(flipped)


def _is_degenerate(path: str) -> bool:
    import nibabel as nib

    return bool(np.allclose(nib.load(path).affine, np.eye(4)))


def find_anchors(data_root: str):
    """Subjects holding both a usable-header scan and a degenerate-affine scan."""
    grouped = collections.defaultdict(lambda: {"usable": [], "degenerate": []})
    for record in discover_scans(data_root):
        bucket = "degenerate" if _is_degenerate(record.source_path) else "usable"
        grouped[record.subject_id][bucket].append(record)
    return {
        subject: value for subject, value in grouped.items()
        if value["usable"] and value["degenerate"]
    }


def _strip(data: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Skull-strip one oriented volume with deepbet.

    Registration to the skull-stripped MNI brain needs a stripped moving image:
    handing it an unstripped head scores around 0.15 instead of 0.6 and the
    comparison this module makes becomes meaningless.
    """
    import shutil
    import tempfile

    import nibabel as nib
    from deepbet import run_bet

    workdir = tempfile.mkdtemp(prefix="handed_")
    try:
        raw_path = os.path.join(workdir, "raw.nii.gz")
        brain_path = os.path.join(workdir, "brain.nii.gz")
        nib.save(nib.Nifti1Image(data.astype(np.float32), affine), raw_path)
        run_bet([raw_path], brain_paths=[brain_path], n_dilate=0)
        if not os.path.exists(brain_path):
            return data
        return np.nan_to_num(
            np.asarray(nib.load(brain_path).dataobj, dtype=np.float32)
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _register_reference(record, template, template_mask, static):
    """Register a usable-header scan, which fixes the subject's true handedness."""
    import nibabel as nib

    image = nib.as_closest_canonical(nib.load(record.source_path))
    data = np.nan_to_num(np.asarray(image.dataobj, dtype=np.float32))
    if data.ndim == 4:
        data = data[..., 0]
    brain = _strip(data, image.affine)
    return register_volume_to_mni(brain, image.affine, template, template_mask, static)


def _register_candidate(record, permutation, flips, template, template_mask, static):
    import nibabel as nib

    image = nib.load(record.source_path)
    data = np.nan_to_num(np.asarray(image.dataobj, dtype=np.float32))
    if data.ndim == 4:
        data = data[..., 0]
    zooms = image.header.get_zooms()[:3]
    oriented = apply_orientation(data, permutation, flips)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([float(zooms[axis]) for axis in permutation])
    brain = _strip(oriented, affine)
    return register_volume_to_mni(brain, affine, template, template_mask, static)


def resolve(data_root: str, table_path: str, out_path: str,
            min_reference_qc: float = 0.30) -> dict:
    """Check each fitted orientation against its mirror on the anchor subjects."""
    template, template_mask = _templates()
    static = _scale_to_unit(
        np.asarray(template.dataobj, dtype=np.float32) * template_mask, template_mask
    )
    table = load_table(table_path)
    anchors = find_anchors(data_root)
    print(f"{len(anchors)} anchor subjects, {len(table)} fitted groups")

    votes = collections.defaultdict(list)
    for subject, scans in sorted(anchors.items()):
        reference_record = scans["usable"][0]
        reference, reference_qc = _register_reference(
            reference_record, template, template_mask, static
        )
        if reference_qc < min_reference_qc:
            print(f"  {subject}: reference registration failed "
                  f"(r={reference_qc:.3f}), skipped", flush=True)
            continue

        for record in scans["degenerate"]:
            import nibabel as nib

            shape = nib.load(record.source_path).shape[:3]
            key = group_key(record.series, shape)
            entry = table.get(key)
            if entry is None:
                continue
            permutation, flips = entry
            fitted, _ = _register_candidate(
                record, permutation, flips, template, template_mask, static
            )
            mirrored, _ = _register_candidate(
                record, permutation, mirror(flips), template, template_mask, static
            )
            fitted_score = qc_correlation(fitted, reference, template_mask)
            mirrored_score = qc_correlation(mirrored, reference, template_mask)
            votes[key].append({
                "subject": subject,
                "reference_qc": round(reference_qc, 4),
                "fitted": round(fitted_score, 4),
                "mirrored": round(mirrored_score, 4),
                "prefers_mirror": bool(mirrored_score > fitted_score),
            })
            print(f"  {subject} {key}: fitted={fitted_score:.4f} "
                  f"mirrored={mirrored_score:.4f} "
                  f"-> {'mirror' if mirrored_score > fitted_score else 'fitted'}",
                  flush=True)

    corrected = dict(load_table(table_path))
    with open(table_path, encoding="utf-8") as handle:
        raw_table = json.load(handle)

    return _apply(votes, corrected, raw_table, table_path, out_path)


def population_reference(data_root: str, template, template_mask, static,
                         limit: int, min_qc: float, cache_path: str | None = None):
    """Mean registered brain over subjects whose headers already fix handedness.

    A single brain is nearly symmetric, so one subject cannot settle a mirror.
    Averaged over subjects, the systematic asymmetries survive while individual
    variation cancels, which is what gives the comparison something to grip.
    """
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        print(f"  reusing cached reference over {len(cached['subjects'])} subjects")
        return cached["reference"].astype(np.float32), list(cached["subjects"])

    reference, used = None, []
    for record in discover_scans(data_root):
        if len(used) >= limit:
            break
        if _is_degenerate(record.source_path) or record.subject_id in used:
            continue
        volume, score = _register_reference(record, template, template_mask, static)
        if score < min_qc:
            continue
        reference = volume if reference is None else reference + volume
        used.append(record.subject_id)
        print(f"  reference {record.subject_id} r={score:.3f} "
              f"({len(used)}/{limit})", flush=True)
    if reference is None:
        return None, []
    reference = reference / len(used)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(cache_path, reference=reference,
                            subjects=np.asarray(used))
    return reference, used


def resolve_by_population(data_root: str, table_path: str, out_path: str,
                          groups: list[str], per_group: int = 6,
                          reference_limit: int = 12,
                          min_qc: float = 0.30,
                          cache_path: str | None = None) -> dict:
    """Settle handedness for groups with no anchor subject, against the cohort mean."""
    import nibabel as nib

    template, template_mask = _templates()
    static = _scale_to_unit(
        np.asarray(template.dataobj, dtype=np.float32) * template_mask, template_mask
    )
    reference, used = population_reference(
        data_root, template, template_mask, static, reference_limit, min_qc,
        cache_path=cache_path,
    )
    if reference is None:
        print("no usable-header subject registered; handedness left unresolved")
        return {}

    corrected = dict(load_table(table_path))
    with open(table_path, encoding="utf-8") as handle:
        raw_table = json.load(handle)

    by_group = collections.defaultdict(list)
    for record in discover_scans(data_root):
        if not _is_degenerate(record.source_path):
            continue
        shape = nib.load(record.source_path).shape[:3]
        by_group[group_key(record.series, shape)].append(record)

    votes = collections.defaultdict(list)
    for key in groups:
        entry = corrected.get(key)
        if entry is None:
            continue
        permutation, flips = entry
        seen, sample = set(), []
        for record in by_group.get(key, []):
            if record.subject_id in seen:
                continue
            seen.add(record.subject_id)
            sample.append(record)
            if len(sample) >= per_group:
                break

        fitted_sum = mirrored_sum = None
        for record in sample:
            fitted, _ = _register_candidate(
                record, permutation, flips, template, template_mask, static
            )
            mirrored, _ = _register_candidate(
                record, permutation, mirror(flips), template, template_mask, static
            )
            fitted_sum = fitted if fitted_sum is None else fitted_sum + fitted
            mirrored_sum = mirrored if mirrored_sum is None else mirrored_sum + mirrored
        if fitted_sum is None:
            continue

        fitted_score = qc_correlation(fitted_sum, reference, template_mask)
        mirrored_score = qc_correlation(mirrored_sum, reference, template_mask)
        votes[key].append({
            "subject": f"population mean of {len(sample)} subjects",
            "reference_qc": None,
            "fitted": round(fitted_score, 4),
            "mirrored": round(mirrored_score, 4),
            "prefers_mirror": bool(mirrored_score > fitted_score),
        })
        print(f"  {key}: fitted={fitted_score:.4f} mirrored={mirrored_score:.4f} "
              f"-> {'mirror' if mirrored_score > fitted_score else 'fitted'}",
              flush=True)

    report = _apply(votes, corrected, raw_table, table_path, out_path,
                    evidence="population mean")
    report["_reference_subjects"] = used
    return report


def _apply(votes, corrected, raw_table, table_path, out_path,
           evidence: str = "anchor subjects") -> dict:
    """Write the resolved handedness back into the orientation table."""
    report = {}

    for key, entries in votes.items():
        mirror_votes = sum(1 for e in entries if e["prefers_mirror"])
        margin = float(np.mean([abs(e["fitted"] - e["mirrored"]) for e in entries]))
        flip_needed = mirror_votes > len(entries) / 2
        report[key] = {
            "anchors": len(entries),
            "mirror_votes": mirror_votes,
            "mean_margin": round(margin, 4),
            "mirror_applied": flip_needed,
            "evidence": entries,
        }
        if flip_needed:
            permutation, flips = corrected[key]
            raw_table[key]["flips"] = [bool(f) for f in mirror(flips)]
            raw_table[key]["handedness"] = f"mirrored against {evidence}"
        else:
            raw_table[key]["handedness"] = f"confirmed against {evidence}"

    for key in raw_table:
        raw_table[key].setdefault("handedness", "unresolved: no anchor subject")

    with open(table_path, "w", encoding="utf-8") as handle:
        json.dump(raw_table, handle, indent=2)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    unresolved = [k for k, v in raw_table.items() if v["handedness"].startswith("unres")]
    if unresolved:
        print(f"unresolved handedness for {len(unresolved)} groups: {unresolved}")
    return report


def enforce_determinant(table_path: str, reference_key: str) -> dict:
    """Give every group the handedness of an anchored group.

    The map from stored array order to RAS is fixed by the DICOM-to-NIfTI
    conversion, so its determinant — the one bit the registration score cannot
    read — should be the same for every group the same converter produced.
    Groups whose fitted determinant disagrees with the anchored group's are
    mirrored to match, and the change is recorded per group so a reader can see
    which orientations rest on measurement and which on this consistency rule.
    """
    with open(table_path, encoding="utf-8") as handle:
        table = json.load(handle)
    reference = table.get(reference_key)
    if reference is None:
        return {}
    target = int(reference["determinant"])

    changed = {}
    for key, entry in table.items():
        if key == reference_key:
            continue
        if str(entry.get("handedness", "")).startswith(("confirmed", "mirrored")):
            continue
        if int(entry["determinant"]) == target:
            entry["handedness"] = (
                f"consistent with {reference_key} (determinant {target:+d})"
            )
            continue
        entry["flips"] = [bool(f) for f in mirror(entry["flips"])]
        entry["determinant"] = target
        entry["handedness"] = (
            f"mirrored for consistency with {reference_key} (determinant {target:+d})"
        )
        changed[key] = entry["flips"]

    with open(table_path, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2)
    return changed


def unresolved_groups(table_path: str) -> list[str]:
    with open(table_path, encoding="utf-8") as handle:
        table = json.load(handle)
    return [k for k, v in table.items()
            if not str(v.get("handedness", "")).startswith(("confirmed", "mirrored"))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="resolve left-right handedness")
    parser.add_argument("--data-root", default="..")
    parser.add_argument("--table", default=os.path.join("data", "orientation_table.json"))
    parser.add_argument("--out", default=os.path.join("data", "handedness_report.json"))
    parser.add_argument("--population-fallback", action="store_true",
                        help="settle groups without an anchor against the cohort mean")
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--reference-limit", type=int, default=12)
    parser.add_argument("--groups", nargs="*", default=None,
                        help="restrict the population fallback to these groups")
    parser.add_argument("--reference-cache",
                        default=os.path.join("data", "handedness_reference.npz"))
    parser.add_argument("--skip-anchors", action="store_true")
    parser.add_argument("--enforce-determinant", default=None,
                        help="propagate this group's handedness to the rest")
    args = parser.parse_args(argv)

    report = {} if args.skip_anchors else resolve(args.data_root, args.table, args.out)
    if args.enforce_determinant:
        changed = enforce_determinant(args.table, args.enforce_determinant)
        print(f"determinant consistency mirrored {len(changed)} groups: "
              f"{sorted(changed)}")
        report = {"anchors": report, "determinant_changed": changed}
    remaining = args.groups or unresolved_groups(args.table)
    if args.population_fallback and remaining:
        print(f"population fallback for {len(remaining)} groups: {remaining}")
        fallback = resolve_by_population(
            args.data_root, args.table, args.out.replace(".json", "_population.json"),
            remaining, per_group=args.per_group,
            reference_limit=args.reference_limit,
            cache_path=args.reference_cache,
        )
        report = {"anchors": report, "population": fallback}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
