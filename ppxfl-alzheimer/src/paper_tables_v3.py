"""
paper_tables_v3.py — generate the paper's v3 tables from the results summary.

Tables are written into marker-delimited blocks in ``paper.tex``:

    % BEGIN AUTO:v3-centralised
    ... generated rows ...
    % END AUTO:v3-centralised

Nothing between the markers is hand-edited, which is what lets the
number-coherence check mean something: a figure in the prose and the
corresponding entry in the results summary cannot drift apart without this
module being re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import re

LABEL_SET_NAMES = {
    "cn-mci-ad": "CN / MCI / AD",
    "cn-ad": "CN vs AD",
    "cn-mci": "CN vs MCI",
    "mci-ad": "MCI vs AD",
}
MODEL_NAMES = {
    "logreg": "Logistic regression",
    "svm": "RBF SVM",
    "lda": "Shrinkage LDA",
    "rf": "Random forest",
    "roi_logreg": "MNI region model",
    "resnet50_head": "ResNet50 head",
    "resnet50_full": "ResNet50, full network",
    "resnet18_full": "ResNet18, 2.5D",
    "resnet18_head": "ResNet18 head",
}
SCHEME_NAMES = {"subject": "subject-disjoint", "site": "site-disjoint"}
CLIENT_NAMES = {
    "natural": "real ADNI sites",
    "iid": "IID",
    "dirichlet": "Dirichlet ($\\alpha$ = 0.5)",
    "none": "---",
}


def cell(entry: dict, scale: float = 1.0, digits: int = 1) -> str:
    """``mean $\\pm$ std`` at fixed precision, or an en-rule when unavailable."""
    if entry is None or entry.get("mean") is None:
        return "---"
    body = f"{entry['mean'] * scale:.{digits}f}"
    if entry.get("std") is None or entry.get("n", 0) < 2:
        return body
    return f"{body} $\\pm$ {entry['std'] * scale:.{digits}f}"


def _rows(conditions, predicate, order_key):
    return sorted((c for c in conditions if predicate(c)), key=order_key)


def table_centralised(summary: dict) -> str:
    lines = []
    for condition in _rows(
        summary["conditions"],
        lambda c: c["family"] == "centralised",
        lambda c: (c["label_set"], c["split_scheme"], c["model"]),
    ):
        lines.append(" & ".join([
            LABEL_SET_NAMES.get(condition["label_set"], condition["label_set"]),
            SCHEME_NAMES.get(condition["split_scheme"], condition["split_scheme"]),
            MODEL_NAMES.get(condition["model"], condition["model"]),
            cell(condition["accuracy"], 100, 1),
            cell(condition["balanced_accuracy"], 100, 1),
            cell(condition["macro_f1"], 1, 3),
            cell(condition["macro_auroc"], 1, 3),
            f"{condition['chance_accuracy'] * 100:.1f}",
            str(condition["n_runs"]),
        ]) + r" \\")
    return "\n".join(lines)


def table_federated(summary: dict) -> str:
    lines = []
    for condition in _rows(
        summary["conditions"],
        lambda c: c["family"] == "federated",
        lambda c: (c["label_set"], c["split_scheme"], c["client_scheme"]),
    ):
        lines.append(" & ".join([
            LABEL_SET_NAMES.get(condition["label_set"], condition["label_set"]),
            SCHEME_NAMES.get(condition["split_scheme"], condition["split_scheme"]),
            CLIENT_NAMES.get(condition["client_scheme"], condition["client_scheme"]),
            cell(condition["accuracy"], 100, 1),
            cell(condition["balanced_accuracy"], 100, 1),
            cell(condition["macro_f1"], 1, 3),
            cell(condition["macro_auroc"], 1, 3),
            str(condition["n_runs"]),
        ]) + r" \\")
    return "\n".join(lines)


def table_privacy(summary: dict) -> str:
    lines = []
    for condition in _rows(
        summary["conditions"],
        lambda c: c["family"] == "privacy",
        lambda c: (c["label_set"], c["target_epsilon"] or 0.0),
    ):
        epsilon = condition["target_epsilon"]
        lines.append(" & ".join([
            LABEL_SET_NAMES.get(condition["label_set"], condition["label_set"]),
            "non-private" if epsilon is None else f"{epsilon:g}",
            "---" if epsilon is None
            else f"{condition.get('noise_multiplier', 0.0):.2f}",
            cell(condition["accuracy"], 100, 1),
            cell(condition["balanced_accuracy"], 100, 1),
            cell(condition["macro_f1"], 1, 3),
            cell(condition.get("explanation_agreement"), 1, 3),
            cell(condition.get("medial_temporal_share"), 100, 1),
            str(condition["n_runs"]),
        ]) + r" \\")
    return "\n".join(lines)


def table_dimension_law(law: dict) -> str:
    lines = []
    for row in sorted(law["rows"], key=lambda r: (r["target_epsilon"], r["dimension"])):
        lines.append(" & ".join([
            f"{row['target_epsilon']:g}",
            MODEL_NAMES.get(row["model"], row["model"].replace("_", r"\_")),
            f"{row['dimension']:,}".replace(",", r"\,"),
            f"{row['noise_multiplier']:.2f}",
            f"{row['expected_noise_norm']:.1f}",
            f"{row['max_signal_norm']:.1f}",
            f"{row['worst_case_ratio']:.2f}",
        ]) + r" \\")
    return "\n".join(lines)


def table_orientation(table: dict) -> str:
    """One row per measured orientation group, with its evidence."""
    lines = []
    for key, entry in sorted(table.items(), key=lambda kv: -kv[1]["scans_in_group"]):
        family, shape = key.split("|", 1)
        handedness = str(entry.get("handedness", "unresolved"))
        handedness = handedness.replace("MPRAGE|", "").replace("_", r"\_")
        dimensions = shape.replace("x", r" \times ")
        lines.append(" & ".join([
            f"{family} ${dimensions}$",
            str(entry["scans_in_group"]),
            "(" + ", ".join(str(a) for a in entry["permutation"]) + ")",
            "(" + ", ".join("T" if f else "F" for f in entry["flips"]) + ")",
            handedness,
        ]) + r" \\")
    return "\n".join(lines)


PREDICTOR_NAMES = {
    "sex": "Sex only",
    "age": "Age only",
    "sex+age": "Sex + age",
    "imaging": "Imaging (MNI regions)",
    "imaging+demographics": "Imaging + demographics",
}


def table_confounds(report: dict) -> str:
    """Demographics-only baselines beside the imaging model and its strata."""
    label_sets = ["cn-mci-ad", "cn-ad"]
    end = r" \\"

    def cell(label_set, predictor):
        entry = report["results"].get(label_set, {}).get(predictor)
        if not entry:
            return "---"
        return f"{entry['balanced_accuracy']['mean'] * 100:.1f}"

    lines = []
    for predictor, name in PREDICTOR_NAMES.items():
        lines.append(" & ".join([name] + [cell(s, predictor) for s in label_sets]) + end)

    lines.append(r"\midrule")
    for stratum, label in (("F", "Imaging, female only"),
                           ("M", "Imaging, male only")):
        values = []
        for label_set in label_sets:
            entry = report["stratified"].get(label_set, {}).get(stratum)
            if not entry:
                values.append("---")
            else:
                mean = entry["balanced_accuracy"]["mean"] * 100
                values.append(f"{mean:.1f} ($n$ = {entry['n']})")
        lines.append(" & ".join([label] + values) + end)
    return "\n".join(lines)


def replace_block(text: str, name: str, body: str) -> tuple[str, bool]:
    pattern = re.compile(
        rf"(% BEGIN AUTO:{re.escape(name)}\n).*?(% END AUTO:{re.escape(name)})",
        re.DOTALL,
    )
    if not pattern.search(text):
        return text, False
    return pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text), True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="generate v3 paper tables")
    parser.add_argument("--summary",
                        default=os.path.join("results_v3", "results_summary_v3.json"))
    parser.add_argument("--law",
                        default=os.path.join("results_v3", "metrics", "dimension_law.json"))
    parser.add_argument("--confounds",
                        default=os.path.join("results_v3", "metrics",
                                             "confounds.json"))
    parser.add_argument("--orientation",
                        default=os.path.join("data", "orientation_table.json"))
    parser.add_argument("--paper", default=os.path.join("..", "paper.tex"))
    args = parser.parse_args(argv)

    with open(args.summary, encoding="utf-8") as handle:
        summary = json.load(handle)

    blocks = {
        "v3-centralised": table_centralised(summary),
        "v3-federated": table_federated(summary),
        "v3-privacy": table_privacy(summary),
    }
    if os.path.exists(args.law):
        with open(args.law, encoding="utf-8") as handle:
            blocks["v3-dimension-law"] = table_dimension_law(json.load(handle))
    if os.path.exists(args.confounds):
        with open(args.confounds, encoding="utf-8") as handle:
            blocks["v3-confounds-rows"] = table_confounds(json.load(handle))
    if os.path.exists(args.orientation):
        with open(args.orientation, encoding="utf-8") as handle:
            blocks["v3-orientation"] = table_orientation(json.load(handle))

    with open(args.paper, encoding="utf-8") as handle:
        text = handle.read()

    written, missing = [], []
    for name, body in blocks.items():
        text, ok = replace_block(text, name, body)
        (written if ok else missing).append(name)

    with open(args.paper, "w", encoding="utf-8") as handle:
        handle.write(text)

    print(f"wrote {len(written)} table blocks: {', '.join(written) or 'none'}")
    if missing:
        print("markers not found in the paper for: " + ", ".join(missing))
        print("add e.g.\n  % BEGIN AUTO:v3-centralised\n  % END AUTO:v3-centralised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
