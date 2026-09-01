"""Tests for the demographic confound analysis.

The module's job is to say how much of an imaging accuracy the cohort's
demographics could have produced on their own, so the tests build cohorts where
that answer is known in advance: one where sex carries the whole signal, and
one where it carries none.
"""

import csv
import json

import numpy as np
import pytest

from src.confounds_v3 import (
    _fit_predict,
    _metrics,
    build_designs,
    evaluate_matrix,
    load_demographics,
    run,
)
from src.confounds_v3 import main as confounds_main


def _export(path, records):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject ID", "Sex", "Research Group", "Visit", "Age",
                         "Description"])
        for subject, sex, group, age in records:
            writer.writerow([subject, sex, group, "sc", age, "MP-RAGE"])


def _cohort(tmp_path, sex_predicts_diagnosis: bool, n_per_class=60, n_features=6):
    """A synthetic cohort with separable imaging and controllable sex balance."""
    rng = np.random.default_rng(0)
    X, y, subjects, sites, records = [], [], [], [], []
    index = 0
    for klass in range(3):
        centre = np.zeros(n_features)
        centre[klass] = 3.0
        for i in range(n_per_class):
            subject = f"{index // 12 + 1:03d}_S_{7000 + index:04d}"
            if sex_predicts_diagnosis:
                sex = "F" if klass == 0 else "M"
            else:
                sex = "F" if i % 2 == 0 else "M"
            X.append(rng.normal(centre, 1.0))
            y.append(klass)
            subjects.append(subject)
            sites.append(f"{index % 12 + 1:03d}")
            records.append((subject, sex, ("CN", "MCI", "AD")[klass], 65 + i % 20))
            index += 1

    features = tmp_path / "features.npz"
    np.savez(features, X=np.asarray(X), y=np.asarray(y, dtype=int),
             subjects=np.asarray(subjects, dtype=object),
             sites=np.asarray(sites, dtype=object),
             feature_names=np.asarray([f"f{i}" for i in range(n_features)],
                                      dtype=object))
    export = tmp_path / "export.csv"
    _export(export, records)
    return features, export


def test_load_demographics_tolerates_a_missing_age(tmp_path):
    path = tmp_path / "export.csv"
    _export(path, [("001_S_0001", "F", "CN", ""), ("001_S_0002", "M", "AD", 71)])
    demographics = load_demographics(str(path))
    assert demographics["001_S_0001"]["age"] is None
    assert demographics["001_S_0002"]["age"] == pytest.approx(71.0)


def test_metrics_report_nan_auroc_when_a_class_is_absent():
    y = np.zeros(6, dtype=int)
    score = np.tile([0.6, 0.4], (6, 1))
    assert np.isnan(_metrics(y, y, score, 2)["macro_auroc"])


def test_fit_predict_recovers_a_separable_split():
    rng = np.random.default_rng(1)
    train_X = np.vstack([rng.normal(-3, 0.5, (40, 2)), rng.normal(3, 0.5, (40, 2))])
    train_y = np.array([0] * 40 + [1] * 40)
    test_X = np.array([[-3.0, -3.0], [3.0, 3.0]])
    pred, score = _fit_predict(train_X, train_y, test_X, seed=0, n_classes=2)
    assert list(pred) == [0, 1]
    assert score.shape == (2, 2)


def test_build_designs_encodes_sex_and_stacks_the_imaging_block():
    features = np.arange(6, dtype=float).reshape(3, 2)
    designs = build_designs(features, ["F", "M", "F"], [70.0, 80.0, 90.0])
    assert designs["sex"].ravel().tolist() == [1.0, 0.0, 1.0]
    assert designs["age"].ravel().tolist() == [70.0, 80.0, 90.0]
    assert designs["sex+age"].shape == (3, 2)
    assert designs["imaging+demographics"].shape == (3, 4)
    np.testing.assert_array_equal(designs["imaging"], features)


def test_evaluate_matrix_summarises_over_seeds(tmp_path):
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(-2, 0.5, (30, 3)), rng.normal(2, 0.5, (30, 3))])
    y = np.array([0] * 30 + [1] * 30)
    sites = [f"{i % 6 + 1:03d}" for i in range(60)]
    result = evaluate_matrix(X, y, sites, "cn-ad", "subject", 3, [0, 1])
    assert result["balanced_accuracy"]["mean"] > 0.9
    assert result["balanced_accuracy"]["std"] >= 0.0


def test_sex_baseline_beats_chance_when_the_cohort_is_confounded(tmp_path):
    features, export = _cohort(tmp_path, sex_predicts_diagnosis=True)
    report = run(str(features), str(export), ["cn-mci-ad"], "subject", 3, [0])
    sex_only = report["results"]["cn-mci-ad"]["sex"]["balanced_accuracy"]["mean"]
    assert sex_only > 1 / 3


def test_sex_baseline_sits_at_chance_when_the_cohort_is_balanced(tmp_path):
    features, export = _cohort(tmp_path, sex_predicts_diagnosis=False)
    report = run(str(features), str(export), ["cn-mci-ad"], "subject", 3, [0])
    results = report["results"]["cn-mci-ad"]
    assert results["sex"]["balanced_accuracy"]["mean"] <= 1 / 3 + 0.05
    # The imaging signal survives regardless, which is the point of the check.
    assert results["imaging"]["balanced_accuracy"]["mean"] > 0.8


def test_stratified_analysis_holds_sex_constant(tmp_path):
    features, export = _cohort(tmp_path, sex_predicts_diagnosis=False)
    report = run(str(features), str(export), ["cn-mci-ad"], "subject", 3, [0])
    strata = report["stratified"]["cn-mci-ad"]
    assert set(strata) == {"F", "M"}
    assert strata["F"]["n"] + strata["M"]["n"] == report["n_subjects"]
    assert strata["F"]["balanced_accuracy"]["mean"] > 0.8


def test_a_stratum_too_small_to_evaluate_is_omitted(tmp_path):
    features, export = _cohort(tmp_path, sex_predicts_diagnosis=True)
    report = run(str(features), str(export), ["cn-ad"], "subject", 3, [0])
    # Every AD subject is male and every CN subject female, so the cn-ad
    # comparison has one class per stratum and neither stratum is usable.
    assert report["stratified"]["cn-ad"] == {}


def test_subjects_without_demographics_are_excluded_and_reported(tmp_path, capsys):
    features, export = _cohort(tmp_path, sex_predicts_diagnosis=False)
    rows = list(csv.DictReader(open(export, encoding="utf-8-sig")))
    with open(export, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows[:-4]:
            writer.writerow(row)

    report = run(str(features), str(export), ["cn-ad"], "subject", 3, [0])
    assert report["n_subjects"] == len(rows) - 4
    assert "demographics available for" in capsys.readouterr().out


def test_cli_writes_the_report(tmp_path):
    features, export = _cohort(tmp_path, sex_predicts_diagnosis=False)
    out = tmp_path / "nested" / "confounds.json"
    assert confounds_main([
        "--features", str(features), "--demographics", str(export),
        "--out", str(out), "--label-sets", "cn-ad", "--n-splits", "3",
        "--seeds", "0",
    ]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) == {"n_subjects", "cohort_composition", "results", "stratified"}
    assert "CN-F" in payload["cohort_composition"]


def test_rows_without_a_subject_identifier_are_ignored(tmp_path):
    path = tmp_path / "export.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject ID", "Sex", "Research Group", "Visit", "Age",
                         "Description"])
        writer.writerow(["", "F", "CN", "sc", "70", "MP-RAGE"])
        writer.writerow(["001_S_0001", "F", "CN", "sc", "70", "MP-RAGE"])
        writer.writerow(["001_S_0001", "M", "AD", "m12", "71", "MP-RAGE"])
    demographics = load_demographics(str(path))
    assert list(demographics) == ["001_S_0001"]
    assert demographics["001_S_0001"]["sex"] == "F"


def test_multiclass_auroc_is_nan_when_a_class_never_appears():
    """One-vs-rest AUROC is undefined if y_true has fewer classes than columns."""
    y = np.array([0, 1, 0, 1])
    score = np.tile([0.5, 0.3, 0.2], (4, 1))
    assert np.isnan(_metrics(y, y, score, 3)["macro_auroc"])


def test_multiclass_auroc_is_reported_when_every_class_appears():
    y = np.array([0, 1, 2, 0, 1, 2])
    score = np.eye(3)[y].astype(float)
    assert _metrics(y, y, score, 3)["macro_auroc"] == pytest.approx(1.0)
