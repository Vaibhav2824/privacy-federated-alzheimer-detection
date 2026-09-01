"""Tests for the cohort expansion selector.

The expansion exists to correct a sex imbalance, so the properties under test
are that it recognises structural T1 series, that it does not re-request
subjects already on disk, and that what it asks for moves each cell towards its
target rather than simply adding more of whatever is plentiful.
"""

import collections
import csv
import json
import os

import pytest

from src.cohort_v4 import (
    age_band,
    existing_subjects,
    is_structural,
    load_export,
    select_balanced,
    summarise,
)
from src.cohort_v4 import main as cohort_main


@pytest.mark.parametrize("description", [
    "MPRAGE", "MP-RAGE", "MP_RAGE", "Accelerated Sagittal MPRAGE (MSV21)",
    "Sag IR-SPGR", "Accelerated Sag IR-FSPGR", "T1 Repeat",
])
def test_structural_series_are_recognised_across_naming_conventions(description):
    assert is_structural(description)


@pytest.mark.parametrize("description", [
    "3 Plane Localizer", "Field Mapping", "B1-Calibration Body",
    "Axial MB rsfMRI (Eyes Open)", "Axial DTI", "3D ASL perfusion",
    # Acquired in the same session as the structural scan and named after it.
    "MPRAGE SENSE localizer", "smartbrain",
])
def test_companion_series_are_not_mistaken_for_the_structural_scan(description):
    assert not is_structural(description)


def _export(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject ID", "Sex", "Research Group", "Visit", "Age",
                         "Description"])
        writer.writerows(rows)


def test_load_export_counts_scans_and_drops_non_structural_rows(tmp_path):
    path = tmp_path / "export.csv"
    _export(path, [
        ["001_S_0001", "F", "CN", "sc", "72.4", "MP-RAGE"],
        ["001_S_0001", "F", "CN", "m12", "73.4", "MPRAGE"],
        ["001_S_0001", "F", "CN", "sc", "72.4", "3 Plane Localizer"],
        ["002_S_0002", "M", "AD", "sc", "not-a-number", "MPRAGE"],
        ["003_S_0003", "M", "MCI", "sc", "80.0", "Field Mapping"],
    ])
    subjects = load_export(str(path))

    assert set(subjects) == {"001_S_0001", "002_S_0002"}
    assert subjects["001_S_0001"]["n_scans"] == 2
    assert subjects["001_S_0001"]["age"] == pytest.approx(72.4)
    # An age that will not parse leaves the subject selectable but unbanded.
    assert subjects["002_S_0002"]["age"] is None


def test_existing_subjects_reads_the_three_class_folders(tmp_path):
    for folder, subject in (("AD-150", "001_S_0001"), ("CN-150", "002_S_0002")):
        os.makedirs(tmp_path / folder / "ADNI" / subject)
    os.makedirs(tmp_path / "MCI-150")  # present but never populated
    assert existing_subjects(str(tmp_path)) == {"001_S_0001", "002_S_0002"}


def test_existing_subjects_is_empty_for_a_fresh_tree(tmp_path):
    assert existing_subjects(str(tmp_path)) == set()


def _candidates(per_cell=20):
    """A candidate pool with plenty of every cell, spread across age bands."""
    out, index = {}, 0
    for group in ("CN", "MCI", "AD"):
        for sex in ("F", "M"):
            for i in range(per_cell):
                subject = f"{index // 10 + 1:03d}_S_{5000 + index:04d}"
                out[subject] = {
                    "subject_id": subject, "group": group, "sex": sex,
                    "age": 65 + (i % 20), "visit": "sc", "n_scans": 1,
                }
                index += 1
    return out


def test_selection_fills_each_cell_to_the_target():
    chosen = select_balanced(_candidates(), collections.Counter(), per_group=20)
    counts = collections.Counter((r["group"], r["sex"]) for r in chosen)
    assert set(counts.values()) == {10}


def test_selection_asks_only_for_what_the_cohort_is_short_of():
    """A cell already at target contributes nothing to the request."""
    current = collections.Counter({("CN", "F"): 10, ("AD", "M"): 4})
    chosen = select_balanced(_candidates(), current, per_group=20)
    counts = collections.Counter((r["group"], r["sex"]) for r in chosen)
    assert ("CN", "F") not in counts
    assert counts[("AD", "M")] == 6
    assert counts[("CN", "M")] == 10


def test_selection_takes_what_exists_when_a_cell_cannot_be_filled():
    candidates = {k: v for k, v in _candidates().items()
                  if not (v["group"] == "AD" and v["sex"] == "F")}
    candidates.update({
        f"099_S_{9000 + i:04d}": {"subject_id": f"099_S_{9000 + i:04d}",
                                  "group": "AD", "sex": "F", "age": 70,
                                  "visit": "sc", "n_scans": 1}
        for i in range(3)
    })
    chosen = select_balanced(candidates, collections.Counter(), per_group=20)
    counts = collections.Counter((r["group"], r["sex"]) for r in chosen)
    assert counts[("AD", "F")] == 3


def test_selection_spreads_across_age_bands():
    chosen = select_balanced(_candidates(), collections.Counter(), per_group=8)
    bands = {age_band(r["age"]) for r in chosen if r["group"] == "CN" and r["sex"] == "F"}
    assert len(bands) > 1


def test_records_of_unknown_group_or_sex_are_never_selected():
    candidates = _candidates()
    candidates["100_S_0001"] = {"subject_id": "100_S_0001", "group": "SMC",
                                "sex": "F", "age": 70, "visit": "sc", "n_scans": 1}
    candidates["100_S_0002"] = {"subject_id": "100_S_0002", "group": "CN",
                                "sex": "U", "age": 70, "visit": "sc", "n_scans": 1}
    chosen = {r["subject_id"] for r in select_balanced(candidates,
                                                       collections.Counter(), 20)}
    assert "100_S_0001" not in chosen
    assert "100_S_0002" not in chosen


def test_summarise_reports_composition_and_mean_age():
    records = [
        {"subject_id": "a", "group": "CN", "sex": "F", "age": 70},
        {"subject_id": "b", "group": "CN", "sex": "M", "age": 80},
        {"subject_id": "c", "group": "AD", "sex": "M", "age": None},
    ]
    summary = summarise(records)
    assert summary["n"] == 3
    assert summary["by_group"] == {"CN": 2, "AD": 1}
    assert summary["by_group_sex"]["CN-F"] == 1
    assert summary["mean_age"] == {"CN": 75.0}


def test_cli_writes_a_reproducible_request(tmp_path, capsys):
    export = tmp_path / "export.csv"
    rows = []
    index = 0
    for group in ("CN", "MCI", "AD"):
        for sex in ("F", "M"):
            for _ in range(6):
                rows.append([f"{index // 10 + 1:03d}_S_{6000 + index:04d}", sex,
                             group, "sc", str(65 + index % 20), "MP-RAGE"])
                index += 1
    _export(export, rows)

    # One subject is already on disk, so the request must exclude it.
    already = rows[0][0]
    os.makedirs(tmp_path / "CN-150" / "ADNI" / already)
    out = tmp_path / "request" / "cohort.json"

    assert cohort_main(["--export", str(export), "--data-root", str(tmp_path),
                        "--per-group", "8", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert already not in payload["subject_ids"]
    assert payload["n_requested"] == len(payload["subject_ids"])
    assert payload["requested"]["by_group_sex"]["CN-F"] == 3
    assert "requesting" in capsys.readouterr().out


@pytest.mark.parametrize("age, expected", [(60, 0), (72, 1), (77, 2), (85, 3)])
def test_age_band_partitions_the_range(age, expected):
    assert age_band(age) == expected


def test_age_band_places_a_missing_age_outside_the_bands():
    """A subject whose age will not parse is still selectable, just unbanded."""
    assert age_band(None) == len(("", "", "", ""))


def test_age_band_clamps_an_age_beyond_the_last_band():
    assert age_band(10_000) == 3
