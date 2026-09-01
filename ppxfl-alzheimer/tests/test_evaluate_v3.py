"""Tests for the v3 centralised evaluation."""

import numpy as np
import pytest

from src.evaluate_v3 import LABEL_SETS, _bootstrap, _metrics, build_model, run_configuration


@pytest.fixture
def cohort():
    """Three separable classes over 150 subjects spread across 10 sites."""
    rng = np.random.default_rng(0)
    per_class, features = 50, 20
    blocks, labels = [], []
    for klass in range(3):
        centre = np.zeros(features)
        centre[klass * 4:(klass + 1) * 4] = 2.0
        blocks.append(rng.normal(centre, 1.0, size=(per_class, features)))
        labels.append(np.full(per_class, klass))
    X = np.vstack(blocks)
    y = np.concatenate(labels)
    sites = [f"{(i % 10) + 1:03d}" for i in range(len(y))]
    return X, y, sites


def test_metrics_are_perfect_for_a_perfect_prediction():
    y = np.array([0, 1, 2, 0, 1, 2])
    score = np.eye(3)[y].astype(float)
    result = _metrics(y, y, score, 3)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["macro_auroc"] == pytest.approx(1.0)


def test_balanced_accuracy_penalises_majority_class_collapse():
    """The failure mode DP produced in the previous study: high accuracy, no recall."""
    y = np.array([0] * 8 + [1] + [2])
    always_zero = np.zeros_like(y)
    score = np.tile([0.9, 0.05, 0.05], (len(y), 1))
    result = _metrics(y, always_zero, score, 3)
    assert result["accuracy"] == pytest.approx(0.8)
    assert result["balanced_accuracy"] == pytest.approx(1 / 3)
    assert result["macro_f1"] < 0.35


def test_auroc_is_undefined_when_a_class_is_absent():
    y = np.array([0, 0, 0])
    score = np.tile([0.5, 0.3, 0.2], (3, 1))
    assert np.isnan(_metrics(y, y, score, 3)["macro_auroc"])


def test_binary_auroc_uses_the_positive_column():
    y = np.array([0, 0, 1, 1])
    score = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    assert _metrics(y, np.array([0, 0, 1, 1]), score, 2)["macro_auroc"] == pytest.approx(1.0)


def test_bootstrap_returns_intervals_that_bracket_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 90)
    pred = y.copy()
    pred[:20] = (pred[:20] + 1) % 3
    score = np.eye(3)[pred].astype(float)
    intervals = _bootstrap(y, pred, score, 3, n=200, seed=0)
    point = _metrics(y, pred, score, 3)["accuracy"]
    low, high = intervals["accuracy_ci95"]
    assert low <= point <= high


def test_bootstrap_skips_degenerate_resamples():
    """A single-class resample cannot be scored and must not raise."""
    y = np.array([0, 0, 0, 1])
    pred = y.copy()
    score = np.eye(2)[pred].astype(float)
    intervals = _bootstrap(y, pred, score, 2, n=50, seed=0)
    assert "accuracy_ci95" in intervals


@pytest.mark.parametrize("name", ["logreg", "svm", "lda", "rf", "ensemble"])
def test_every_model_fits_and_returns_probabilities(name, cohort):
    X, y, _ = cohort
    model = build_model(name, seed=0)
    model.fit(X, y)
    probability = model.predict_proba(X)
    assert probability.shape == (len(y), 3)
    assert np.allclose(probability.sum(axis=1), 1.0)


def test_build_model_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("nonsense", seed=0)


def test_run_configuration_beats_chance_on_separable_data(cohort):
    X, y, sites = cohort
    result = run_configuration(X, y, sites, "logreg", "cn-mci-ad", "subject", 5, 42)
    assert result["overall"]["accuracy"] > result["chance_accuracy"]
    assert result["n_subjects"] == len(y)
    assert result["uniform_chance"] == pytest.approx(1 / 3)
    assert len(result["per_fold"]) == 5


def test_run_configuration_restricts_to_the_requested_label_set(cohort):
    X, y, sites = cohort
    result = run_configuration(X, y, sites, "logreg", "cn-ad", "subject", 5, 42)
    assert result["class_names"] == ["CN", "AD"]
    assert result["n_subjects"] == int(np.isin(y, (0, 2)).sum())
    assert result["uniform_chance"] == pytest.approx(0.5)


def test_run_configuration_supports_site_disjoint_splits(cohort):
    X, y, sites = cohort
    result = run_configuration(X, y, sites, "lda", "cn-mci-ad", "site", 5, 42)
    assert result["split_scheme"] == "site"
    assert "accuracy_ci95" in result["overall"]


def test_label_sets_cover_the_pairwise_contrasts():
    assert set(LABEL_SETS) == {"cn-mci-ad", "cn-ad", "cn-mci", "mci-ad"}
    for spec in LABEL_SETS.values():
        assert len(spec["classes"]) == len(spec["names"])


def test_load_features_round_trips_a_saved_matrix(tmp_path, cohort):
    """The loader must return exactly what src.features wrote."""
    from src.evaluate_v3 import load_features
    from src.features import save_features

    X, y, sites = cohort
    names = [f"gm::R{i}" for i in range(X.shape[1])]
    subjects = [f"00{i % 9}_S_{1000 + i}" for i in range(len(y))]
    path = tmp_path / "features.npz"
    save_features(str(path), X.astype(np.float32), y, subjects, sites, names)

    loaded_X, loaded_y, loaded_subjects, loaded_sites, loaded_names = \
        load_features(str(path))
    assert loaded_X.shape == X.shape
    assert np.allclose(loaded_X, X, atol=1e-5)
    assert list(loaded_y) == list(y)
    assert loaded_subjects == subjects
    assert loaded_sites == sites
    assert loaded_names == names


def test_evaluate_cli_writes_one_file_per_configuration(tmp_path, cohort, capsys):
    from src.evaluate_v3 import main
    from src.features import save_features

    X, y, sites = cohort
    names = [f"gm::R{i}" for i in range(X.shape[1])]
    subjects = [f"00{i % 9}_S_{1000 + i}" for i in range(len(y))]
    features = tmp_path / "features.npz"
    save_features(str(features), X.astype(np.float32), y, subjects, sites, names)

    out = tmp_path / "metrics"
    assert main([
        "--features", str(features), "--out-dir", str(out),
        "--models", "lda", "--label-sets", "cn-ad",
        "--schemes", "subject", "--seeds", "42", "--n-splits", "3",
    ]) == 0
    written = sorted(p.name for p in out.iterdir())
    assert written == ["centralised_summary.json", "lda_cn-ad_subject_s42.json"]
    assert "lda_cn-ad_subject_s42" in capsys.readouterr().out


def _demographics_export(path, subjects, labels, confounded: bool):
    import csv

    names = ("CN", "MCI", "AD")
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Subject ID", "Sex", "Research Group", "Visit", "Age",
                         "Description"])
        for i, (subject, label) in enumerate(zip(subjects, labels)):
            sex = ("F" if label == 0 else "M") if confounded else ("F", "M")[i % 2]
            writer.writerow([subject, sex, names[int(label)], "sc", 65 + i % 20,
                             "MP-RAGE"])


@pytest.fixture
def cohort_with_demographics(tmp_path, cohort):
    from src.evaluate_v3 import restrict_to_cohort  # noqa: F401

    X, y, sites = cohort
    subjects = [f"{i // 15 + 1:03d}_S_{8000 + i:04d}" for i in range(len(y))]
    export = tmp_path / "export.csv"
    _demographics_export(export, subjects, y, confounded=False)
    return X, y, subjects, sites, str(export)


def test_cohort_all_passes_every_subject_through(cohort_with_demographics):
    from src.evaluate_v3 import restrict_to_cohort

    X, y, subjects, sites, export = cohort_with_demographics
    out = restrict_to_cohort(X, y, subjects, sites, "all", "", export)
    assert out[0].shape == X.shape
    assert out[2] == subjects


def test_full_cohort_keeps_only_subjects_with_demographics(cohort_with_demographics,
                                                           tmp_path):
    from src.evaluate_v3 import restrict_to_cohort

    X, y, subjects, sites, _ = cohort_with_demographics
    export = tmp_path / "partial.csv"
    _demographics_export(export, subjects[:100], y[:100], confounded=False)

    _, y_out, subjects_out, sites_out = restrict_to_cohort(
        X, y, subjects, sites, "full", "", str(export)
    )
    assert subjects_out == subjects[:100]
    assert len(y_out) == len(sites_out) == 100


def test_balanced_cohort_is_smaller_and_sex_balanced(cohort_with_demographics,
                                                     tmp_path):
    from src.analysis_cohort import describe, load_demographics
    from src.evaluate_v3 import restrict_to_cohort

    X, y, subjects, sites, _ = cohort_with_demographics
    balanced_export = tmp_path / "balanced.csv"
    _demographics_export(balanced_export, subjects, y, confounded=False)
    _, y_out, subjects_out, _ = restrict_to_cohort(
        X, y, subjects, sites, "balanced", "", str(balanced_export)
    )
    demographics = load_demographics(str(balanced_export))
    summary = describe(subjects_out, y_out, demographics,
                       list(range(len(subjects_out))))
    assert summary["sex_only_rate"] <= 1 / 3 + 1e-4


def test_a_missing_demographics_file_leaves_the_cohort_untouched(cohort_with_demographics):
    from src.evaluate_v3 import restrict_to_cohort

    X, y, subjects, sites, _ = cohort_with_demographics
    out = restrict_to_cohort(X, y, subjects, sites, "full", "", "does-not-exist.csv")
    assert out[2] == subjects


def test_an_impossible_balance_falls_back_but_says_so(cohort_with_demographics,
                                                      tmp_path, capsys):
    """A silent fallback would file a full-cohort number under a balanced heading."""
    from src.evaluate_v3 import restrict_to_cohort

    X, y, subjects, sites, _ = cohort_with_demographics
    export = tmp_path / "confounded.csv"
    # CN entirely female and the others entirely male: no cell can be balanced.
    _demographics_export(export, subjects, y, confounded=True)

    out = restrict_to_cohort(X, y, subjects, sites, "balanced", "", str(export))
    assert out[2] == subjects
    assert "selected no subjects" in capsys.readouterr().out


def test_an_unknown_cohort_name_is_refused(cohort_with_demographics):
    from src.evaluate_v3 import restrict_to_cohort

    X, y, subjects, sites, export = cohort_with_demographics
    with pytest.raises(ValueError, match="unknown cohort"):
        restrict_to_cohort(X, y, subjects, sites, "everything", "", export)
