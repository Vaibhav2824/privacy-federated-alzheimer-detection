"""export_site: the site's data file is built only from recorded aggregates."""

import json

import pytest

from src import export_site


def _summary(path, conditions):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"conditions": conditions}), encoding="utf-8")


def _central(model, acc, n=100, label_set="cn-mci-ad"):
    return {"family": "centralised", "model": model, "label_set": label_set,
            "split_scheme": "subject", "client_scheme": "none", "target_epsilon": None,
            "n_runs": 3, "n_subjects": n, "chance_accuracy": 0.34,
            "accuracy": {"mean": acc, "std": 0.01}, "explanation_agreement": None}


def _private(eps, agreement):
    return {"family": "privacy", "model": "roi_logreg", "label_set": "cn-ad",
            "split_scheme": "subject", "client_scheme": "natural", "target_epsilon": eps,
            "n_runs": 3, "n_subjects": 100, "chance_accuracy": 0.5, "noise_multiplier": 1.8,
            "accuracy": {"mean": 0.8, "std": 0.01},
            "explanation_agreement": {"mean": agreement, "std": 0.1}}


@pytest.fixture
def root(tmp_path):
    (tmp_path / "results" / "metrics").mkdir(parents=True)
    (tmp_path / "results" / "metrics" / "all_experiments_results.csv").write_text(
        "experiment,accuracy\nresnet50_centralised,0.91\nresnet50_fedavg_K4,0.97\n", encoding="utf-8")
    v2 = tmp_path / "results_v2" / "metrics"
    v2.mkdir(parents=True)
    for i, acc in enumerate((0.3, 0.4)):
        (v2 / f"resnet50_centralised_f{i}_v2_metrics.json").write_text(json.dumps({"accuracy": acc}))
    _summary(tmp_path / "results_v3" / "results_summary_v3.json",
             [_central("ensemble", 0.51, 297), _private(10.0, 0.06)])
    _summary(tmp_path / "results_v4" / "results_summary_v4.json", [_central("ensemble", 0.525, 760)])
    _summary(tmp_path / "results_v5" / "results_summary_v5.json",
             [_central("ensemble", 0.51, 799), _private(10.0, 0.44)])
    v6 = tmp_path / "results_v6"
    _summary(v6 / "results_summary_v6_imaging_only.json",
             [_central("ensemble_cv", 0.586, 799), _private(10.0, 0.2)])
    _summary(v6 / "results_summary_v6.json", [_central("ensemble_cv", 0.584, 799), _private(1.0, 0.14)])
    m = v6 / "metrics"
    m.mkdir()
    (m / "dp_cn-mci-ad_eps1.0_s42.json").write_text(json.dumps({
        "clip_norm": 1.0, "rounds": 60, "local_epochs": 3, "subject_sample_rate": 0.5, "delta": 1e-3,
        "partition": {"n_clients": 2, "clients": [
            {"n_subjects": 9, "sites": ["005"], "class_counts": {"0": 1, "2": 8}, "label_entropy": 0.5},
            {"n_subjects": 23, "sites": ["002", "003"], "class_counts": {"0": 13, "1": 6, "2": 4},
             "label_entropy": 0.98}]}}))
    (m / "mia_v3.json").write_text(json.dumps({"runs": [
        {"label_set": "cn-mci-ad", "target_epsilon": None, "attack_auroc": 0.6},
        {"label_set": "cn-mci-ad", "target_epsilon": None, "attack_auroc": 0.58},
        {"label_set": "cn-mci-ad", "target_epsilon": 1, "attack_auroc": 0.53}]}))
    (m / "confounds.json").write_text(json.dumps({"n_subjects": 677}))
    (m / "dimension_law.json").write_text(json.dumps({"rows": [{"model": "roi_logreg", "dimension": 405}]}))
    return tmp_path


def test_build_exports_aggregates_only(root):
    data = export_site.build(str(root))
    families = [c["family"] for c in data["conditions"]]
    # imaging-only centralised rows, then the covariate run with its centralised rows relabelled
    assert families == ["centralised", "centralised+cov", "privacy"]
    assert data["conditions"][0]["accuracy"] == 0.586
    assert "explanation_agreement" not in data["conditions"][0]
    assert data["attack_auroc"] == {"cn-mci-ad|none": pytest.approx(0.59), "cn-mci-ad|1.0": 0.53}
    assert data["partition"]["n_sites"] == 3
    assert [c["n_subjects"] for c in data["partition"]["clients"]] == [23, 9]
    assert data["partition"]["clients"][1]["class_counts"] == [1, 0, 8]
    assert "sites" not in data["partition"]["clients"][0]          # no site identifiers leave
    h = data["history"]
    assert h["v1"]["accuracy"] == 0.97 and h["v2"]["accuracy"] == pytest.approx(0.35)
    assert h["v3"]["agreement_eps10"] == 0.06 and h["v5"]["agreement_eps10"] == 0.44
    assert h["v6"] == {"accuracy": 0.586, "n_subjects": 799}


def test_headline_refuses_a_summary_without_the_model(root):
    with pytest.raises(KeyError, match="no centralised"):
        export_site._headline(str(root / "results_v4" / "results_summary_v4.json"), model="svm")


def test_main_writes_the_file(root, tmp_path, capsys):
    out = tmp_path / "site.json"
    assert export_site.main(["--root", str(root), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["cohort"] == "v6"
    assert "3 conditions, 2 clients" in capsys.readouterr().out
