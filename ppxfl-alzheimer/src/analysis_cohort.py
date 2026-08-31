"""
analysis_cohort.py — define which subjects an analysis runs on, and why.

Two cohorts are defined over the same registered subjects, and both are
reported.  They answer different questions and neither alone is sufficient.

``full``
    Every subject that passed registration QC.  Maximum power, and it uses the
    data that exists rather than a subset chosen after the fact.  Its weakness
    is that ADNI's diagnosis groups differ in sex composition, so part of any
    measured accuracy could be a sex classifier rather than an anatomical one.

``balanced``
    A subset in which every diagnosis contributes equally and, within each
    diagnosis, each sex contributes equally, drawn evenly across age bands.
    Sex and diagnosis are orthogonal by construction, so a sex-only rule scores
    exactly chance and there is no confound left to adjust for.  It costs
    subjects, and the subset is chosen without reference to any image or
    outcome, only to the demographic table.

Reporting both is the point.  A result that holds on the balanced cohort is not
a demographic artefact; a result that also holds on the full cohort is not an
artefact of which subjects the balancing happened to keep.  Where they
disagree, the balanced cohort is the one to believe about the effect and the
full cohort is the one to believe about precision.

Selection is deterministic given the seed, uses no image data and no label
beyond the diagnosis being balanced, and is therefore not a form of fitting to
the outcome.
"""

from __future__ import annotations

import collections
import csv
import random

GROUPS = ("CN", "MCI", "AD")
GROUP_INDEX = {"CN": 0, "MCI": 1, "AD": 2}
AGE_BANDS = ((0, 70), (70, 75), (75, 80), (80, 200))


def load_demographics(path: str) -> dict:
    """Subject -> sex and age from an IDA advanced-search export."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            subject = row.get("Subject ID")
            if not subject or subject in out:
                continue
            try:
                age = float(row["Age"])
            except (TypeError, ValueError, KeyError):
                age = None
            out[subject] = {"sex": row.get("Sex"), "age": age}
    return out


def age_band(age) -> int:
    if age is None:
        return len(AGE_BANDS)
    for index, (low, high) in enumerate(AGE_BANDS):
        if low <= age < high:
            return index
    return len(AGE_BANDS) - 1


def balanced_indices(subjects, labels, demographics, seed: int = 42) -> list[int]:
    """Indices of a diagnosis- and sex-balanced, age-spread subset.

    The cell size is set by the scarcest (diagnosis, sex) cell, so the result is
    the largest balanced subset the cohort supports.
    """
    rng = random.Random(seed)
    cells: dict[tuple, list[int]] = collections.defaultdict(list)
    for index, (subject, label) in enumerate(zip(subjects, labels)):
        record = demographics.get(subject)
        if record is None or record["sex"] not in ("M", "F"):
            continue
        group = GROUPS[int(label)]
        cells[(group, record["sex"])].append(index)

    required = [(g, s) for g in GROUPS for s in ("F", "M")]
    if any(not cells[key] for key in required):
        return []
    per_cell = min(len(cells[key]) for key in required)

    chosen: list[int] = []
    for key in required:
        by_band: dict[int, list[int]] = collections.defaultdict(list)
        for index in cells[key]:
            by_band[age_band(demographics[subjects[index]]["age"])].append(index)
        for band in by_band.values():
            rng.shuffle(band)
        picked, bands = [], sorted(by_band)
        while len(picked) < per_cell and any(by_band[b] for b in bands):
            for band in bands:
                if len(picked) >= per_cell:
                    break
                if by_band[band]:
                    picked.append(by_band[band].pop())
        chosen.extend(picked)
    return sorted(chosen)


def describe(subjects, labels, demographics, indices) -> dict:
    """Composition of a cohort, including what a demographics-only rule could score."""
    counts = collections.Counter()
    ages = collections.defaultdict(list)
    for index in indices:
        record = demographics.get(subjects[index])
        if record is None:
            continue
        group = GROUPS[int(labels[index])]
        counts[(group, record["sex"])] += 1
        if record["age"] is not None:
            ages[group].append(record["age"])

    by_group = collections.Counter()
    for (group, _sex), n in counts.items():
        by_group[group] += n
    total = sum(by_group.values()) or 1

    # Best accuracy obtainable from sex alone: for each sex, always guess that
    # sex's most common diagnosis.
    by_sex: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for (group, sex), n in counts.items():
        by_sex[sex][group] += n
    sex_only = sum(max(c.values()) for c in by_sex.values()) / total

    return {
        "n": len(indices),
        "by_group": dict(by_group),
        "by_group_sex": {f"{g}-{s}": n for (g, s), n in sorted(counts.items())},
        "mean_age": {g: round(sum(v) / len(v), 1) for g, v in sorted(ages.items()) if v},
        "majority_class_rate": round(max(by_group.values()) / total, 4),
        "sex_only_rate": round(sex_only, 4),
    }


def build(features_path: str, demographics_path: str, seed: int = 42) -> dict:
    """Both cohorts and their composition, from a features file."""
    from .evaluate_v3 import load_features

    _, labels, subjects, _, _ = load_features(features_path)
    demographics = load_demographics(demographics_path)

    full = [i for i, s in enumerate(subjects) if s in demographics]
    balanced = balanced_indices(subjects, labels, demographics, seed=seed)
    return {
        "seed": seed,
        "full": {"indices": full,
                 **describe(subjects, labels, demographics, full)},
        "balanced": {"indices": balanced,
                     **describe(subjects, labels, demographics, balanced)},
    }


def main(argv=None) -> int:
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="define the analysis cohorts")
    parser.add_argument("--features", default=os.path.join("data", "mni2mm", "features.npz"))
    parser.add_argument("--demographics", default=os.path.join("data", "ida_search_v4.csv"))
    parser.add_argument("--out", default=os.path.join("data", "analysis_cohorts.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cohorts = build(args.features, args.demographics, args.seed)
    for name in ("full", "balanced"):
        summary = {k: v for k, v in cohorts[name].items() if k != "indices"}
        print(f"{name}:")
        print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(cohorts, handle, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
