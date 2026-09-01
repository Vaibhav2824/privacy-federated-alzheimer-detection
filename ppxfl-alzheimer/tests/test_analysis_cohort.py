"""Tests for the analysis cohort definitions.

The balanced cohort exists to remove the sex confound by construction, so the
property worth testing is not that some subset comes back but that a sex-only
rule scores exactly chance on it.
"""

import csv
import json

import numpy as np
import pytest

from src.analysis_cohort import (
    GROUPS,
    age_band,
    balanced_indices,
    build,
    describe,
    load_demographics,
)
from src.analysis_cohort import main as cohort_main


def _write_export(path, records):
    """An IDA advanced-search export with one row per subject."""
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject ID", "Sex", "Research Group", "Visit", "Age",
                         "Description"])
        for subject, sex, group, age in records:
            writer.writerow([subject, sex, group, "sc", age, "MP-RAGE"])


@pytest.fixture
def cohort(tmp_path):
    """A cohort whose sex composition differs sharply between diagnoses.

    CN is three-quarters female and AD three-quarters male, which is the shape
    of the real confound: a rule reading sex alone beats the chance rate.
    """
    records, subjects, labels = [], [], []
    plan = {"CN": ("F", "F", "F", "M"), "MCI": ("F", "M", "M", "M"),
            "AD": ("M", "M", "M", "F")}
    index = 0
    for group, sexes in plan.items():
        for _repeat in range(5):
            for sex in sexes:
                subject = f"{index // 10 + 1:03d}_S_{4000 + index:04d}"
                age = 65 + (index % 20)
                records.append((subject, sex, group, age))
                subjects.append(subject)
                labels.append(GROUPS.index(group))
                index += 1
    export = tmp_path / "export.csv"
    _write_export(export, records)
    return export, subjects, np.asarray(labels)


def test_load_demographics_keeps_the_first_row_per_subject(tmp_path):
    export = tmp_path / "export.csv"
    _write_export(export, [("001_S_0001", "F", "CN", 70),
                           ("001_S_0001", "M", "AD", 90),
                           ("001_S_0002", "M", "AD", "n/a")])
    demographics = load_demographics(export)
    assert demographics["001_S_0001"] == {"sex": "F", "age": 70.0}
    # An unparseable age is recorded as missing rather than dropping the subject.
    assert demographics["001_S_0002"] == {"sex": "M", "age": None}


@pytest.mark.parametrize("age, expected", [
    (60, 0), (69.9, 0), (70, 1), (74, 1), (75, 2), (79, 2), (80, 3), (200, 3),
])
def test_age_band_partitions_the_range(age, expected):
    assert age_band(age) == expected


def test_age_band_places_a_missing_age_outside_the_bands():
    assert age_band(None) == 4


def test_balanced_cohort_makes_sex_uninformative(cohort):
    export, subjects, labels = cohort
    demographics = load_demographics(export)
    index = balanced_indices(subjects, labels, demographics, seed=42)

    summary = describe(subjects, labels, demographics, index)
    assert summary["sex_only_rate"] == pytest.approx(1 / 3, abs=5e-5)
    assert summary["majority_class_rate"] == pytest.approx(1 / 3, abs=5e-5)
    # Every (diagnosis, sex) cell contributes the same number of subjects.
    assert len(set(summary["by_group_sex"].values())) == 1


def test_full_cohort_retains_the_confound(cohort):
    """The comparison only means something if the full cohort is still skewed."""
    export, subjects, labels = cohort
    demographics = load_demographics(export)
    full = list(range(len(subjects)))
    assert describe(subjects, labels, demographics, full)["sex_only_rate"] > 1 / 3


def test_balanced_selection_is_deterministic_given_the_seed(cohort):
    export, subjects, labels = cohort
    demographics = load_demographics(export)
    first = balanced_indices(subjects, labels, demographics, seed=7)
    assert first == balanced_indices(subjects, labels, demographics, seed=7)


def test_balancing_is_impossible_when_a_cell_is_empty(cohort):
    """An all-female AD group leaves nothing to balance against, so nothing is returned."""
    export, subjects, labels = cohort
    demographics = load_demographics(export)
    for subject, label in zip(subjects, labels):
        if GROUPS[label] == "AD":
            demographics[subject]["sex"] = "F"
    assert balanced_indices(subjects, labels, demographics) == []


def test_subjects_with_unknown_sex_are_skipped(cohort):
    export, subjects, labels = cohort
    demographics = load_demographics(export)
    demographics[subjects[0]]["sex"] = "U"
    assert subjects.index(subjects[0]) not in balanced_indices(subjects, labels, demographics)


def test_describe_ignores_a_subject_with_no_demographic_record(cohort):
    export, subjects, labels = cohort
    demographics = load_demographics(export)
    del demographics[subjects[0]]
    summary = describe(subjects, labels, demographics, list(range(len(subjects))))
    assert summary["n"] == len(subjects)
    assert sum(summary["by_group"].values()) == len(subjects) - 1


def test_missing_ages_do_not_break_the_age_summary(cohort):
    export, subjects, labels = cohort
    demographics = load_demographics(export)
    for record in demographics.values():
        record["age"] = None
    summary = describe(subjects, labels, demographics, list(range(len(subjects))))
    assert summary["mean_age"] == {}


def _write_features(path, subjects, labels):
    np.savez(
        path,
        X=np.zeros((len(subjects), 3), dtype=np.float64),
        y=labels.astype(int),
        subjects=np.asarray(subjects, dtype=object),
        sites=np.asarray([s.split("_")[0] for s in subjects], dtype=object),
        feature_names=np.asarray(["a", "b", "c"], dtype=object),
    )


def test_build_reports_both_cohorts(tmp_path, cohort):
    export, subjects, labels = cohort
    features = tmp_path / "features.npz"
    _write_features(features, subjects, labels)

    cohorts = build(str(features), str(export), seed=42)
    assert cohorts["full"]["n"] == len(subjects)
    assert cohorts["balanced"]["n"] < cohorts["full"]["n"]
    assert cohorts["balanced"]["sex_only_rate"] == pytest.approx(1 / 3, abs=5e-5)


def test_cli_writes_both_cohorts(tmp_path, cohort, capsys):
    export, subjects, labels = cohort
    features = tmp_path / "features.npz"
    _write_features(features, subjects, labels)
    out = tmp_path / "nested" / "cohorts.json"

    assert cohort_main(["--features", str(features), "--demographics", str(export),
                        "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) == {"seed", "full", "balanced"}
    assert "balanced" in capsys.readouterr().out
