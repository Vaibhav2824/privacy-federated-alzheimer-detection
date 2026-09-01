"""Tests for archive ingestion.

Routing is the part that can silently corrupt the cohort: a scan filed under
the wrong diagnosis is worse than a scan that never arrived, so a subject whose
label cannot be established is counted and left alone rather than guessed at.
"""

import csv
import json
import os
import zipfile

import pytest

from src.ingest_v4 import (
    extract,
    find_archives,
    labels_from_export,
    requested_subjects,
    route,
)
from src.ingest_v4 import main as ingest_main

LABELS = {"001_S_0001": "CN", "002_S_0002": "AD", "003_S_0003": "MCI"}


def _export(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject ID", "Sex", "Research Group", "Visit", "Age",
                         "Description"])
        writer.writerows(rows)


def _archive(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)


def test_labels_from_export_keeps_the_first_group_and_ignores_other_cohorts(tmp_path):
    path = tmp_path / "export.csv"
    _export(path, [
        ["001_S_0001", "F", "CN", "sc", "70", "MP-RAGE"],
        ["001_S_0001", "F", "AD", "m24", "72", "MP-RAGE"],
        ["004_S_0004", "M", "SMC", "sc", "68", "MP-RAGE"],
    ])
    labels = labels_from_export(str(path))
    assert labels == {"001_S_0001": "CN"}


def test_labels_from_a_missing_export_are_empty(tmp_path):
    assert labels_from_export(str(tmp_path / "absent.csv")) == {}


def test_requested_subjects_reads_the_cohort_request(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"subject_ids": ["001_S_0001", "002_S_0002"]}),
                    encoding="utf-8")
    assert requested_subjects(str(path)) == {"001_S_0001", "002_S_0002"}
    assert requested_subjects(str(tmp_path / "absent.json")) == set()


def test_find_archives_selects_matching_zips_and_skips_metadata(tmp_path):
    for name in ("PPXFL_v4_part1.zip", "PPXFL_v4_part2.zip",
                 "PPXFL_v4_metadata.zip", "unrelated.zip", "PPXFL_v4_notes.txt"):
        (tmp_path / name).write_bytes(b"")
    found = [os.path.basename(p) for p in find_archives(str(tmp_path), r"PPXFL_v4")]
    assert found == ["PPXFL_v4_part1.zip", "PPXFL_v4_part2.zip"]


def test_route_files_a_scan_under_its_diagnosis_and_keeps_the_adni_layout():
    member = "PPXFL/ADNI/002_S_0002/MP-RAGE/2011-01-01_00_00_00.0/I12345/scan.nii"
    folder, relative = route(member, LABELS)
    assert folder == "AD-150"
    assert relative == "ADNI/002_S_0002/MP-RAGE/2011-01-01_00_00_00.0/I12345/scan.nii"


def test_route_handles_windows_separators():
    member = r"ADNI\001_S_0001\MPRAGE\2010-01-01\I1\scan.nii"
    folder, relative = route(member, LABELS)
    assert folder == "CN-150"
    assert relative.startswith("ADNI/001_S_0001/")


def test_route_falls_back_to_the_trailing_path_when_there_is_no_adni_root():
    member = "003_S_0003/MPRAGE/2010-01-01/I1/scan.nii"
    folder, relative = route(member, LABELS)
    assert folder == "MCI-150"
    assert relative == member


def test_route_declines_a_member_with_no_subject_identifier():
    assert route("ADNI/readme.txt", LABELS) is None


def test_route_declines_a_subject_with_no_label():
    assert route("ADNI/009_S_9999/MPRAGE/d/I1/scan.nii", LABELS) is None


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "PPXFL_v4_part1.zip"
    _archive(path, {
        "ADNI/001_S_0001/MPRAGE/2010-01-01/I1/scan.nii": b"cn-scan",
        "ADNI/002_S_0002/MPRAGE/2010-01-01/I2/scan.nii": b"ad-scan",
        "ADNI/009_S_9999/MPRAGE/2010-01-01/I3/scan.nii": b"unlabelled",
        "ADNI/001_S_0001/MPRAGE/2010-01-01/I1/notes.txt": b"ignored",
        "ADNI/001_S_0001/": b"",
    })
    return path


def test_extract_routes_each_scan_to_its_class_folder(archive, tmp_path):
    root = tmp_path / "cohort"
    stats = extract(str(archive), str(root), LABELS, dry_run=False)

    assert stats["extracted"] == 2
    assert stats["unlabelled"] == 1
    assert stats["subjects"] == {"001_S_0001", "002_S_0002"}
    scan = root / "CN-150" / "ADNI" / "001_S_0001" / "MPRAGE" / "2010-01-01" / "I1" / "scan.nii"
    assert scan.read_bytes() == b"cn-scan"
    # Non-image members are never written.
    assert not (scan.parent / "notes.txt").exists()


def test_extract_is_idempotent(archive, tmp_path):
    root = tmp_path / "cohort"
    extract(str(archive), str(root), LABELS, dry_run=False)
    second = extract(str(archive), str(root), LABELS, dry_run=False)
    assert second["extracted"] == 0
    assert second["skipped"] == 2


def test_dry_run_writes_nothing(archive, tmp_path):
    root = tmp_path / "cohort"
    stats = extract(str(archive), str(root), LABELS, dry_run=True)
    assert stats["extracted"] == 2
    assert not root.exists()


def test_cli_reports_requested_subjects_that_the_archive_did_not_contain(
        archive, tmp_path, capsys):
    export = tmp_path / "export.csv"
    _export(export, [[s, "F", g, "sc", "70", "MP-RAGE"] for s, g in LABELS.items()])
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"subject_ids": sorted(LABELS)}), encoding="utf-8")

    assert ingest_main([
        "--download-dir", str(tmp_path), "--data-root", str(tmp_path / "cohort"),
        "--export", str(export), "--request", str(request), "--dry-run",
    ]) == 0
    out = capsys.readouterr().out
    # 003_S_0003 was requested by the design but is absent from the archive.
    assert "requested but absent: 1" in out
    assert "003_S_0003" in out


def test_cli_fails_when_no_archive_matches(tmp_path, capsys):
    assert ingest_main(["--download-dir", str(tmp_path), "--pattern", "NOPE",
                        "--export", str(tmp_path / "absent.csv"),
                        "--request", str(tmp_path / "absent.json")]) == 1
    assert "no archives matching" in capsys.readouterr().out


def test_a_relative_path_without_the_subject_is_still_extracted(tmp_path):
    """The subject can sit above the ADNI root, leaving it out of the kept path."""
    path = tmp_path / "PPXFL_v4_odd.zip"
    _archive(path, {"001_S_0001/ADNI/scan.nii": b"cn-scan"})
    stats = extract(str(path), str(tmp_path / "cohort"), LABELS, dry_run=False)
    assert stats["extracted"] == 1
    assert stats["subjects"] == set()
    assert (tmp_path / "cohort" / "CN-150" / "ADNI" / "scan.nii").exists()


def test_cli_says_nothing_about_a_request_it_was_not_given(archive, tmp_path, capsys):
    export = tmp_path / "export.csv"
    _export(export, [[s, "F", g, "sc", "70", "MP-RAGE"] for s, g in LABELS.items()])
    assert ingest_main([
        "--download-dir", str(tmp_path), "--data-root", str(tmp_path / "cohort"),
        "--export", str(export), "--request", str(tmp_path / "absent.json"),
        "--dry-run",
    ]) == 0
    assert "requested but absent" not in capsys.readouterr().out


def test_cli_reports_a_fully_satisfied_request(archive, tmp_path, capsys):
    export = tmp_path / "export.csv"
    _export(export, [[s, "F", g, "sc", "70", "MP-RAGE"] for s, g in LABELS.items()])
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"subject_ids": ["001_S_0001", "002_S_0002"]}),
                       encoding="utf-8")
    assert ingest_main([
        "--download-dir", str(tmp_path), "--data-root", str(tmp_path / "cohort"),
        "--export", str(export), "--request", str(request), "--dry-run",
    ]) == 0
    out = capsys.readouterr().out
    assert "requested but absent: 0" in out
    assert "first few" not in out
