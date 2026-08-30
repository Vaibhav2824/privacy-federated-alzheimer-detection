"""
ingest_v4.py — take a downloaded ADNI collection archive into the cohort.

The IDA advanced download delivers one or more zips whose internal layout is
``ADNI/<subject>/<series>/<date>/<image id>/*.nii``.  The existing cohort is
stored as three class folders with that same layout underneath, so ingestion is
a matter of routing each subject to the folder its diagnosis implies and
leaving the rest of the path intact -- which is what ``preprocess_v3`` already
knows how to read.

Diagnosis comes from the cohort request written by ``cohort_v4``, falling back
to the IDA search export.  A subject whose label cannot be established is left
in place and reported rather than guessed at, since a mislabelled subject is
worse than a missing one.

The script is idempotent: a scan already present at its destination is skipped,
so an interrupted extraction can simply be re-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import zipfile

SUBJECT_RE = re.compile(r"\d{3}_S_\d{4,5}")
CLASS_DIRS = {"CN": "CN-150", "MCI": "MCI-150", "AD": "AD-150"}


def labels_from_export(path: str) -> dict:
    """Subject -> diagnosis, from an IDA advanced-search CSV export."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            subject = row.get("Subject ID")
            group = row.get("Research Group")
            if subject and group in CLASS_DIRS:
                out.setdefault(subject, group)
    return out


def requested_subjects(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return set(json.load(handle).get("subject_ids", []))


def find_archives(download_dir: str, pattern: str) -> list[str]:
    matcher = re.compile(pattern, re.IGNORECASE)
    return sorted(
        os.path.join(download_dir, name)
        for name in os.listdir(download_dir)
        if name.lower().endswith(".zip") and matcher.search(name)
        and "metadata" not in name.lower()
    )


def route(member: str, labels: dict) -> tuple[str, str] | None:
    """Destination class folder and relative path for one archive member."""
    match = SUBJECT_RE.search(member.replace("\\", "/"))
    if match is None:
        return None
    subject = match.group(0)
    group = labels.get(subject)
    if group is None:
        return None
    # Keep everything from the ADNI/ root so the layout matches the cohort.
    parts = member.replace("\\", "/").split("/")
    if "ADNI" in parts:
        relative = "/".join(parts[parts.index("ADNI"):])
    else:
        relative = "/".join(parts[-5:])
    return CLASS_DIRS[group], relative


def extract(archive: str, data_root: str, labels: dict, dry_run: bool) -> dict:
    stats = {"extracted": 0, "skipped": 0, "unlabelled": 0, "subjects": set()}
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            if not member.lower().endswith((".nii", ".nii.gz", ".dcm")):
                continue
            routed = route(member, labels)
            if routed is None:
                stats["unlabelled"] += 1
                continue
            folder, relative = routed
            destination = os.path.join(data_root, folder, relative)
            match = SUBJECT_RE.search(relative)
            if match:
                stats["subjects"].add(match.group(0))
            if os.path.exists(destination):
                stats["skipped"] += 1
                continue
            if dry_run:
                stats["extracted"] += 1
                continue
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with zf.open(member) as source, open(destination, "wb") as target:
                shutil.copyfileobj(source, target)
            stats["extracted"] += 1
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ingest a downloaded ADNI archive")
    parser.add_argument("--download-dir",
                        default=os.path.join(os.path.expanduser("~"), "Downloads"))
    parser.add_argument("--pattern", default=r"PPXFL_v4",
                        help="regex the archive filename must contain")
    parser.add_argument("--data-root", default="..")
    parser.add_argument("--export", default=os.path.join("data", "ida_search_v4.csv"))
    parser.add_argument("--request", default=os.path.join("data", "cohort_v4_request.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    labels = labels_from_export(args.export)
    requested = requested_subjects(args.request)
    print(f"{len(labels)} subjects labelled from the export; "
          f"{len(requested)} requested by the cohort design")

    archives = find_archives(args.download_dir, args.pattern)
    if not archives:
        print(f"no archives matching {args.pattern!r} in {args.download_dir}")
        return 1

    total = {"extracted": 0, "skipped": 0, "unlabelled": 0}
    seen: set[str] = set()
    for archive in archives:
        size = os.path.getsize(archive) / (1024 ** 3)
        print(f"\n{os.path.basename(archive)} ({size:.2f} GB)")
        stats = extract(archive, args.data_root, labels, args.dry_run)
        seen |= stats["subjects"]
        for key in total:
            total[key] += stats[key]
        print(f"  extracted {stats['extracted']}, skipped {stats['skipped']}, "
              f"unlabelled {stats['unlabelled']}, subjects {len(stats['subjects'])}")

    print(f"\ntotal: {total['extracted']} files extracted, {total['skipped']} already "
          f"present, {total['unlabelled']} unlabelled")
    print(f"subjects in archives: {len(seen)}")
    if requested:
        missing = requested - seen
        print(f"requested but absent: {len(missing)}")
        if missing:
            print("  first few:", ", ".join(sorted(missing)[:8]))
    print("\nnext: python run_v3.py   (registration picks up new subjects automatically)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
