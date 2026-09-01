"""Tests for the dashboard export.

The dashboard carries one chance line for the whole table, so the export must
not mix label sets: a 34% chance rule drawn under a binary row would invite
exactly the wrong reading.
"""

import json

import pytest

from src.export_ui_v3 import build, condition_name, to_dashboard_condition
from src.export_ui_v3 import main as export_main


def _condition(**overrides):
    entry = {
        "family": "centralised",
        "model": "ensemble",
        "label_set": "cn-mci-ad",
        "split_scheme": "subject",
        "client_scheme": "none",
        "target_epsilon": None,
        "n_runs": 3,
        "n_subjects": 760,
        "chance_accuracy": 0.34,
        "uniform_chance": 1 / 3,
        "accuracy": {"mean": 0.539, "std": 0.01, "n": 3},
        "balanced_accuracy": {"mean": 0.52, "std": 0.01, "n": 3},
        "macro_f1": {"mean": 0.51, "std": 0.02, "n": 3},
        "macro_auroc": {"mean": 0.70, "std": 0.01, "n": 3},
    }
    entry.update(overrides)
    return entry


def test_condition_name_describes_the_split():
    assert condition_name(_condition(split_scheme="site")) == "site"


def test_condition_name_names_the_client_partition_and_budget():
    name = condition_name(_condition(family="federated", client_scheme="natural"))
    assert name == "subject, natural clients"
    private = condition_name(_condition(family="privacy", target_epsilon=1.0))
    assert private == "subject, eps 1"


def test_private_conditions_carry_the_scope_and_accounted_budget():
    entry = _condition(
        family="privacy", target_epsilon=1.0, perturbed_dimension=1234,
        accounted_epsilon={"mean": 0.97, "std": 0.0, "n": 3},
    )
    out = to_dashboard_condition(entry)
    assert out["method"] == "dpfedavg_userlevel"
    assert out["dp_scope"] == "subject"
    assert out["actual_epsilon_mean"] == pytest.approx(0.97)
    assert out["perturbed_params"] == 1234


def test_non_private_conditions_report_no_budget_or_dimension():
    out = to_dashboard_condition(_condition(family="federated", perturbed_dimension=99))
    assert out["method"] == "fedavg"
    assert out["dp_scope"] is None
    assert out["actual_epsilon_mean"] is None
    # Perturbed dimension is meaningless without a private arm.
    assert out["perturbed_params"] is None


def test_unknown_families_keep_their_own_name():
    assert to_dashboard_condition(_condition(family="something-new"))["method"] == \
        "something-new"


def test_build_keeps_one_label_set_and_sorts_by_accuracy():
    summary = {"conditions": [
        _condition(accuracy={"mean": 0.50, "std": 0.0, "n": 3}, split_scheme="site"),
        _condition(accuracy={"mean": 0.55, "std": 0.0, "n": 3}),
        _condition(label_set="cn-ad", accuracy={"mean": 0.80, "std": 0.0, "n": 3}),
    ]}
    payload = build(summary)
    assert payload["n_conditions"] == 2
    assert [c["accuracy_mean"] for c in payload["conditions"]] == [0.55, 0.50]
    assert payload["chance_accuracy"] == pytest.approx(0.34)
    assert "760 ADNI subjects" in payload["cohort"]


def test_build_drops_a_condition_with_no_measured_accuracy():
    summary = {"conditions": [
        _condition(),
        _condition(accuracy={"mean": None, "std": None, "n": 0}),
    ]}
    assert build(summary)["n_conditions"] == 1


def test_build_refuses_a_label_set_that_is_not_present():
    with pytest.raises(ValueError, match="no conditions"):
        build({"conditions": [_condition()]}, label_set="mci-ad")


def test_cli_writes_the_dashboard_contract(tmp_path, capsys):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"conditions": [_condition()]}), encoding="utf-8")
    out = tmp_path / "public" / "data" / "results_summary.json"

    assert export_main(["--summary", str(summary), "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_conditions"] == 1
    assert "chance" in capsys.readouterr().out
