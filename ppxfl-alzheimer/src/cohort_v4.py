"""
cohort_v4.py — choose which ADNI subjects to add, and on what principle.

The v3 cohort was whatever the earlier download happened to contain, and it
carries a confound that only became visible once demographics were checked:
CN is 66% female while MCI is 62% male, so *sex alone* predicts diagnosis well
above chance on that cohort.  Any accuracy measured there is partly a sex
classifier, and no amount of careful cross-validation removes that — it is a
property of the sample, not of the protocol.

So the expansion is not "download more scans".  It selects subjects to make the
cohort balanced on diagnosis and, within each diagnosis, on sex, and it matches
the age distribution across groups.  A confound removed by design needs no
statistical adjustment afterwards and cannot be argued with by a reviewer.

Selection is deterministic given the seed and the input export, and the chosen
identifiers are written out so the cohort is reproducible for anyone with their
own approved ADNI download.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re

# Structural T1 series.  ADNI names them inconsistently across phases -- MPRAGE,
# MP-RAGE, Accelerated Sagittal MPRAGE, Sag IR-SPGR and so on -- so the match is
# on family rather than on an exact list.
STRUCTURAL = re.compile(r"(MPRAGE|MP-RAGE|MP_RAGE|SPGR|\bT1\b)", re.IGNORECASE)

# Series that are acquired alongside the structural scan but are not it.
NON_STRUCTURAL = re.compile(
    r"(localizer|scout|calibration|field ?map|survey|rsfMRI|fMRI|DTI|ASL|"
    r"perfusion|B1|smartbrain)",
    re.IGNORECASE,
)

GROUPS = ("CN", "MCI", "AD")
AGE_BANDS = ((0, 70), (70, 75), (75, 80), (80, 200))


def is_structural(description: str) -> bool:
    return bool(STRUCTURAL.search(description)) and not NON_STRUCTURAL.search(description)


def load_export(path: str) -> dict:
    """One record per subject that has at least one structural T1."""
    with open(path, encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if is_structural(r["Description"])]

    subjects: dict[str, dict] = {}
    for row in rows:
        subject = row["Subject ID"]
        if subject in subjects:
            subjects[subject]["n_scans"] += 1
            continue
        try:
            age = float(row["Age"])
        except (TypeError, ValueError):
            age = None
        subjects[subject] = {
            "subject_id": subject,
            "group": row["Research Group"],
            "sex": row["Sex"],
            "age": age,
            "visit": row["Visit"],
            "n_scans": 1,
        }
    return subjects


def existing_subjects(data_root: str) -> set[str]:
    """Subjects already downloaded, so the expansion does not re-fetch them."""
    have: set[str] = set()
    for folder in ("AD-150", "CN-150", "MCI-150"):
        path = os.path.join(data_root, folder, "ADNI")
        if os.path.isdir(path):
            have |= set(os.listdir(path))
    return have


def age_band(age) -> int:
    if age is None:
        return len(AGE_BANDS)
    for index, (low, high) in enumerate(AGE_BANDS):
        if low <= age < high:
            return index
    return len(AGE_BANDS) - 1


def select_balanced(candidates: dict, current: collections.Counter, per_group: int,
                    seed: int = 42) -> list[dict]:
    """Pick new subjects so each (group, sex) cell reaches ``per_group`` / 2.

    Within a cell, subjects are drawn evenly across age bands, so the expansion
    does not quietly shift one group older than another while fixing sex.
    """
    rng = random.Random(seed)
    by_cell: dict[tuple, list] = collections.defaultdict(list)
    for record in candidates.values():
        if record["group"] in GROUPS and record["sex"] in ("M", "F"):
            by_cell[(record["group"], record["sex"])].append(record)

    chosen: list[dict] = []
    for group in GROUPS:
        for sex in ("F", "M"):
            target = per_group // 2 - current[(group, sex)]
            if target <= 0:
                continue
            pool = by_cell[(group, sex)]
            by_band: dict[int, list] = collections.defaultdict(list)
            for record in pool:
                by_band[age_band(record["age"])].append(record)
            for band in by_band.values():
                rng.shuffle(band)

            # Round-robin over age bands until the cell is filled.
            picked, bands = [], sorted(by_band)
            while len(picked) < target and any(by_band[b] for b in bands):
                for band in bands:
                    if len(picked) >= target:
                        break
                    if by_band[band]:
                        picked.append(by_band[band].pop())
            chosen.extend(picked)
    return chosen


def summarise(records) -> dict:
    counts = collections.Counter((r["group"], r["sex"]) for r in records)
    ages = collections.defaultdict(list)
    for record in records:
        if record["age"] is not None:
            ages[record["group"]].append(record["age"])
    return {
        "n": len(records),
        "by_group": dict(collections.Counter(r["group"] for r in records)),
        "by_group_sex": {f"{g}-{s}": n for (g, s), n in sorted(counts.items())},
        "mean_age": {g: round(sum(v) / len(v), 1) for g, v in sorted(ages.items()) if v},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="select the cohort expansion")
    parser.add_argument("--export", required=True, help="IDA advanced-search CSV export")
    parser.add_argument("--data-root", default="..")
    parser.add_argument("--per-group", type=int, default=334,
                        help="target subjects per diagnosis in the final cohort")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join("data", "cohort_v4_request.json"))
    args = parser.parse_args(argv)

    subjects = load_export(args.export)
    have = existing_subjects(args.data_root)
    print(f"{len(subjects)} subjects with a structural T1; {len(have)} already downloaded")

    current = collections.Counter()
    for subject_id, record in subjects.items():
        if subject_id in have:
            current[(record["group"], record["sex"])] += 1
    print("current cohort by group and sex:", dict(current))

    candidates = {k: v for k, v in subjects.items() if k not in have}
    chosen = select_balanced(candidates, current, args.per_group, args.seed)

    combined = [subjects[s] for s in have if s in subjects] + chosen
    payload = {
        "per_group_target": args.per_group,
        "seed": args.seed,
        "n_requested": len(chosen),
        "requested": summarise(chosen),
        "resulting_cohort": summarise(combined),
        "subject_ids": sorted(r["subject_id"] for r in chosen),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nrequesting {len(chosen)} new subjects")
    print(json.dumps(payload["requested"], indent=2))
    print("resulting cohort:")
    print(json.dumps(payload["resulting_cohort"], indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
