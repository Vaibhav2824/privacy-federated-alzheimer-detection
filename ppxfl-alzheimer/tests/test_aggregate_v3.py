"""Tests for the v3 result aggregation and the paper tables it feeds."""

import json

import numpy as np
import pytest

from src.aggregate_v3 import _summarise, aggregate, best_of, condition_key, load_runs
from src.aggregate_v3 import main as aggregate_main
from src.paper_tables_v3 import (
    cell,
    replace_block,
    table_centralised,
    table_dimension_law,
    table_federated,
    table_privacy,
)
from src.paper_tables_v3 import main as tables_main


def _run(**overrides):
    payload = {
        "label_set": "cn-mci-ad",
        "split_scheme": "subject",
        "n_subjects": 280,
        "chance_accuracy": 0.36,
        "uniform_chance": 1 / 3,
        "overall": {
            "accuracy": 0.55, "balanced_accuracy": 0.54,
            "macro_f1": 0.52, "macro_auroc": 0.71,
        },
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def results_dir(tmp_path):
    directory = tmp_path / "metrics"
    directory.mkdir()
    for seed, accuracy in ((42, 0.55), (123, 0.57), (2024, 0.53)):
        payload = _run(model="logreg")
        payload["overall"]["accuracy"] = accuracy
        (directory / f"logreg_cn-mci-ad_subject_s{seed}.json").write_text(
            json.dumps(payload), encoding="utf-8")
    for seed in (42, 123):
        payload = _run(client_scheme="natural")
        (directory / f"fed_natural_cn-mci-ad_subject_s{seed}.json").write_text(
            json.dumps(payload), encoding="utf-8")
    for seed in (42, 123):
        payload = _run(client_scheme="natural", target_epsilon=2.0,
                       noise_multiplier=6.34, perturbed_dimension=423,
                       epsilon=2.0,
                       explainability={"agreement_spearman": 0.3,
                                       "medial_temporal_share": 0.11})
        (directory / f"dp_cn-mci-ad_eps2.0_s{seed}.json").write_text(
            json.dumps(payload), encoding="utf-8")
    # Files the loader must ignore.
    (directory / "centralised_summary.json").write_text("[]", encoding="utf-8")
    (directory / "dimension_law.json").write_text('{"rows": []}', encoding="utf-8")
    (directory / "orientation_sources.json").write_text('{"a": "table"}',
                                                        encoding="utf-8")
    return directory


def test_summarise_reports_mean_and_spread():
    result = _summarise([0.4, 0.6, 0.5])
    assert result["mean"] == pytest.approx(0.5)
    assert result["n"] == 3
    assert result["std"] > 0


def test_summarise_of_one_value_has_zero_spread():
    assert _summarise([0.4]) == {"mean": 0.4, "std": 0.0, "n": 1}


def test_summarise_drops_missing_and_nan_values():
    assert _summarise([None, float("nan"), 0.5])["n"] == 1


def test_summarise_of_nothing_is_empty():
    assert _summarise([None, float("nan")]) == {"mean": None, "std": None, "n": 0}


def test_load_runs_skips_summaries_and_files_without_metrics(results_dir):
    runs = load_runs(str(results_dir))
    names = {r["_file"] for r in runs}
    assert "centralised_summary.json" not in names
    assert "dimension_law.json" not in names
    assert "orientation_sources.json" not in names
    assert len(runs) == 7


def test_load_runs_labels_each_family(results_dir):
    families = {r["_file"]: r["_family"] for r in load_runs(str(results_dir))}
    assert families["dp_cn-mci-ad_eps2.0_s42.json"] == "privacy"
    assert families["fed_natural_cn-mci-ad_subject_s42.json"] == "federated"
    assert families["logreg_cn-mci-ad_subject_s42.json"] == "centralised"


def test_condition_key_separates_privacy_budgets():
    private = _run(target_epsilon=2.0, model="roi_logreg")
    private["_family"] = "privacy"
    public = _run(model="roi_logreg")
    public["_family"] = "privacy"
    assert condition_key(private) != condition_key(public)


def test_aggregate_groups_seeds_of_the_same_condition(results_dir):
    summary = aggregate(str(results_dir))
    centralised = [c for c in summary["conditions"] if c["family"] == "centralised"]
    assert len(centralised) == 1
    assert centralised[0]["n_runs"] == 3
    assert centralised[0]["accuracy"]["mean"] == pytest.approx(0.55)


def test_aggregate_carries_privacy_specific_fields(results_dir):
    summary = aggregate(str(results_dir))
    private = [c for c in summary["conditions"] if c["family"] == "privacy"][0]
    assert private["noise_multiplier"] == pytest.approx(6.34)
    assert private["perturbed_dimension"] == 423
    assert private["explanation_agreement"]["mean"] == pytest.approx(0.3)
    assert private["medial_temporal_share"]["mean"] == pytest.approx(0.11)


def test_best_of_ignores_private_conditions(results_dir):
    summary = aggregate(str(results_dir))
    best = best_of(summary, "privacy", "cn-mci-ad", "subject")
    assert best is None


def test_best_of_picks_the_highest_scoring_condition(results_dir):
    summary = aggregate(str(results_dir))
    best = best_of(summary, "centralised", "cn-mci-ad", "subject")
    assert best["model"] == "logreg"


def test_best_of_returns_none_when_nothing_matches(results_dir):
    summary = aggregate(str(results_dir))
    assert best_of(summary, "centralised", "cn-ad", "subject") is None


def test_aggregate_cli_writes_a_summary(results_dir, tmp_path, capsys):
    out = tmp_path / "summary.json"
    assert aggregate_main(["--results-dir", str(results_dir), "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["n_runs"] == 7
    assert "conditions" in capsys.readouterr().out or written["n_conditions"] > 0


def test_aggregate_cli_skips_conditions_without_an_accuracy(tmp_path):
    directory = tmp_path / "metrics"
    directory.mkdir()
    payload = _run(model="logreg")
    payload["overall"]["accuracy"] = None
    (directory / "logreg_cn-mci-ad_subject_s42.json").write_text(
        json.dumps(payload), encoding="utf-8")
    out = tmp_path / "summary.json"
    assert aggregate_main(["--results-dir", str(directory), "--out", str(out)]) == 0


def test_cell_formats_mean_and_spread():
    assert cell({"mean": 0.55, "std": 0.02, "n": 3}, 100, 1) == r"55.0 $\pm$ 2.0"


def test_cell_omits_spread_for_a_single_run():
    assert cell({"mean": 0.55, "std": 0.0, "n": 1}, 100, 1) == "55.0"


def test_cell_of_a_missing_value_is_an_en_rule():
    assert cell(None) == "---"
    assert cell({"mean": None}) == "---"


def test_tables_render_one_row_per_condition(results_dir):
    summary = aggregate(str(results_dir))
    assert table_centralised(summary).count(r"\\") == 1
    assert table_federated(summary).count(r"\\") == 1
    assert table_privacy(summary).count(r"\\") == 1


def test_privacy_table_labels_the_non_private_row(results_dir, tmp_path):
    directory = tmp_path / "m"
    directory.mkdir()
    payload = _run(client_scheme="natural", target_epsilon=None)
    (directory / "dp_cn-mci-ad_nonprivate_s42.json").write_text(
        json.dumps(payload), encoding="utf-8")
    summary = aggregate(str(directory))
    assert "non-private" in table_privacy(summary)


def test_dimension_law_table_lists_every_row():
    law = {"rows": [
        {"model": "roi_logreg", "target_epsilon": 2.0, "dimension": 423,
         "noise_multiplier": 6.34, "expected_noise_norm": 130.4,
         "max_signal_norm": 112.0, "worst_case_ratio": 1.16},
        {"model": "resnet50_full", "target_epsilon": 2.0, "dimension": 23508035,
         "noise_multiplier": 6.34, "expected_noise_norm": 30740.0,
         "max_signal_norm": 112.0, "worst_case_ratio": 274.5},
    ]}
    rendered = table_dimension_law(law)
    assert rendered.count(r"\\") == 2
    assert r"23\,508\,035" in rendered


def test_replace_block_substitutes_between_markers():
    text = "before\n% BEGIN AUTO:x\nold\n% END AUTO:x\nafter\n"
    out, ok = replace_block(text, "x", "new")
    assert ok
    assert "new" in out and "old" not in out
    assert out.startswith("before") and out.rstrip().endswith("after")


def test_replace_block_reports_a_missing_marker():
    out, ok = replace_block("no markers here", "x", "new")
    assert not ok
    assert out == "no markers here"


def test_tables_cli_writes_present_blocks_and_reports_missing(results_dir, tmp_path,
                                                             capsys):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(aggregate(str(results_dir))), encoding="utf-8")
    paper = tmp_path / "paper.tex"
    paper.write_text("% BEGIN AUTO:v3-centralised\n% END AUTO:v3-centralised\n",
                     encoding="utf-8")
    law = tmp_path / "dimension_law.json"
    law.write_text(json.dumps({"rows": []}), encoding="utf-8")

    assert tables_main(["--summary", str(summary_path), "--paper", str(paper),
                        "--law", str(law),
                        "--orientation", str(tmp_path / "absent.json")]) == 0
    output = capsys.readouterr().out
    assert "v3-centralised" in output
    assert "markers not found" in output
    assert "logreg" in paper.read_text(encoding="utf-8").lower() or \
        "Logistic regression" in paper.read_text(encoding="utf-8")


def test_tables_cli_without_a_dimension_law_file(results_dir, tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(aggregate(str(results_dir))), encoding="utf-8")
    paper = tmp_path / "paper.tex"
    paper.write_text("% BEGIN AUTO:v3-privacy\n% END AUTO:v3-privacy\n",
                     encoding="utf-8")
    assert tables_main(["--summary", str(summary_path), "--paper", str(paper),
                        "--law", str(tmp_path / "absent.json"),
                        "--orientation", str(tmp_path / "absent.json")]) == 0
    assert "non-private" in paper.read_text(encoding="utf-8") or \
        "2" in paper.read_text(encoding="utf-8")


def test_aggregated_metrics_are_finite(results_dir):
    summary = aggregate(str(results_dir))
    for condition in summary["conditions"]:
        for key in ("accuracy", "macro_f1"):
            assert np.isfinite(condition[key]["mean"])


def test_tables_cli_reports_nothing_missing_when_every_marker_exists(results_dir,
                                                                    tmp_path, capsys):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(aggregate(str(results_dir))), encoding="utf-8")
    law = tmp_path / "dimension_law.json"
    law.write_text(json.dumps({"rows": []}), encoding="utf-8")
    orientation = tmp_path / "orientation_table.json"
    orientation.write_text(json.dumps({
        "MPRAGE|256x240x160": {
            "permutation": [2, 1, 0], "flips": [True, True, True],
            "scans_in_group": 107,
            "handedness": "confirmed against anchor subjects",
        }
    }), encoding="utf-8")
    confounds = tmp_path / "confounds.json"
    confounds.write_text(json.dumps({
        "results": {
            "cn-mci-ad": {
                name: {"balanced_accuracy": {"mean": value, "std": 0.01}}
                for name, value in (("sex", 0.414), ("age", 0.365),
                                    ("sex+age", 0.411), ("imaging", 0.516),
                                    ("imaging+demographics", 0.537))
            },
        },
        "stratified": {
            "cn-mci-ad": {
                "F": {"balanced_accuracy": {"mean": 0.505, "std": 0.01}, "n": 146},
                "M": {"balanced_accuracy": {"mean": 0.496, "std": 0.01}, "n": 150},
            },
        },
    }), encoding="utf-8")

    paper = tmp_path / "paper.tex"
    paper.write_text("\n".join(
        f"% BEGIN AUTO:{name}\n% END AUTO:{name}"
        for name in ("v3-centralised", "v3-federated", "v3-privacy",
                     "v3-dimension-law", "v3-confounds-rows", "v3-orientation")
    ) + "\n", encoding="utf-8")

    assert tables_main(["--summary", str(summary_path), "--paper", str(paper),
                        "--law", str(law), "--confounds", str(confounds),
                        "--orientation", str(orientation)]) == 0
    output = capsys.readouterr().out
    assert "wrote 6 table blocks" in output
    assert "markers not found" not in output
    assert "Sex only" in paper.read_text(encoding="utf-8")
    assert r"256 \times 240" in paper.read_text(encoding="utf-8")
