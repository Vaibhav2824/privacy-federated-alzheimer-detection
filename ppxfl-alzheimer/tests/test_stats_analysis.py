"""Tests for bootstrap confidence intervals and paired significance testing."""

import json
import os

import numpy as np
import pytest

import stats_analysis as stats_module


class TestSubjectLevelBootstrapCi:
    def test_perfect_predictions_give_a_ci_pinned_at_one(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        subjects = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
        result = stats_module.subject_level_bootstrap_ci(y, y, subjects, n_boot=50, seed=0)
        assert result['accuracy']['mean'] == pytest.approx(1.0)
        assert result['accuracy']['ci_lo'] == pytest.approx(1.0)
        assert result['accuracy']['ci_hi'] == pytest.approx(1.0)

    def test_all_wrong_predictions_give_zero_accuracy(self):
        y_true = np.zeros(6, dtype=int)
        y_pred = np.ones(6, dtype=int)
        subjects = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
        result = stats_module.subject_level_bootstrap_ci(y_true, y_pred, subjects, n_boot=50, seed=0)
        assert result['accuracy']['mean'] == pytest.approx(0.0)

    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 3, 60)
        y_pred = y_true.copy()
        y_pred[:20] = (y_pred[:20] + 1) % 3
        subjects = np.repeat([f's{i}' for i in range(20)], 3)
        result = stats_module.subject_level_bootstrap_ci(y_true, y_pred, subjects, n_boot=200, seed=1)
        acc = result['accuracy']
        assert acc['ci_lo'] <= acc['mean'] <= acc['ci_hi']
        assert acc['ci_lo'] < acc['ci_hi']

    def test_reports_both_accuracy_and_macro_f1(self):
        y = np.array([0, 1, 2, 0])
        subjects = np.array(['a', 'a', 'b', 'b'])
        result = stats_module.subject_level_bootstrap_ci(y, y, subjects, n_boot=20, seed=0)
        assert set(result) == {'accuracy', 'f1_macro', 'n_boot'}
        assert result['n_boot'] == 20

    def test_resamples_subjects_not_slices(self):
        """Two subjects, one wholly right and one wholly wrong: resampling whole
        subjects can only ever produce accuracies of 0, 0.5 or 1."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 2, 2])
        subjects = np.array(['good', 'good', 'bad', 'bad'])
        result = stats_module.subject_level_bootstrap_ci(y_true, y_pred, subjects,
                                                         n_boot=200, seed=0)
        assert result['accuracy']['ci_lo'] in (0.0, 0.5)
        assert result['accuracy']['ci_hi'] in (0.5, 1.0)

    def test_wider_alpha_gives_a_narrower_interval(self):
        rng = np.random.RandomState(2)
        y_true = rng.randint(0, 3, 40)
        y_pred = rng.randint(0, 3, 40)
        subjects = np.repeat([f's{i}' for i in range(20)], 2)
        narrow = stats_module.subject_level_bootstrap_ci(y_true, y_pred, subjects,
                                                         n_boot=200, seed=3, alpha=0.5)
        wide = stats_module.subject_level_bootstrap_ci(y_true, y_pred, subjects,
                                                       n_boot=200, seed=3, alpha=0.01)
        narrow_width = narrow['accuracy']['ci_hi'] - narrow['accuracy']['ci_lo']
        wide_width = wide['accuracy']['ci_hi'] - wide['accuracy']['ci_lo']
        assert narrow_width <= wide_width


class TestPairedWilcoxon:
    def test_identical_series_are_reported_as_no_difference(self):
        result = stats_module.paired_wilcoxon([0.5] * 5, [0.5] * 5, 'a', 'b')
        assert result['p_value'] == 1.0
        assert result['statistic'] == 0.0
        assert result['n_pairs'] == 5

    def test_a_consistent_gap_is_significant(self):
        a = [0.90, 0.91, 0.89, 0.92, 0.88, 0.93, 0.90, 0.91]
        b = [0.40, 0.41, 0.39, 0.42, 0.38, 0.43, 0.40, 0.41]
        result = stats_module.paired_wilcoxon(a, b, 'non-DP', 'DP')
        assert result['significant_at_0.05'] is True
        assert result['mean_a'] > result['mean_b']

    def test_reports_the_comparison_label(self):
        result = stats_module.paired_wilcoxon([1, 2, 3, 4], [1, 3, 2, 5], 'x', 'y')
        assert result['comparison'] == 'x vs y'

    def test_refuses_unpaired_series(self):
        assert 'error' in stats_module.paired_wilcoxon([1, 2, 3], [1, 2], 'a', 'b')

    def test_refuses_fewer_than_two_pairs(self):
        assert 'error' in stats_module.paired_wilcoxon([1], [2], 'a', 'b')


class TestCollectMetrics:
    def test_reads_files_matching_the_pattern(self, tmp_path):
        (tmp_path / 'a_metrics.json').write_text(json.dumps({'accuracy': 0.5}))
        (tmp_path / 'b_metrics.json').write_text(json.dumps({'accuracy': 0.6}))
        (tmp_path / 'c_history.json').write_text('{}')
        rows = stats_module.collect_metrics(str(tmp_path), '*_metrics.json')
        assert len(rows) == 2
        assert all('_path' in row for row in rows)

    def test_skips_unreadable_files(self, tmp_path):
        (tmp_path / 'a_metrics.json').write_text(json.dumps({'accuracy': 0.5}))
        (tmp_path / 'b_metrics.json').write_text('{oops')
        assert len(stats_module.collect_metrics(str(tmp_path), '*_metrics.json')) == 1

    def test_empty_directory_yields_nothing(self, tmp_path):
        assert stats_module.collect_metrics(str(tmp_path), '*_metrics.json') == []


def _metrics(tmp_path, tag, accuracy=0.5, f1=None):
    payload = {'accuracy': accuracy, 'f1_macro': f1 if f1 is not None else accuracy - 0.05}
    (tmp_path / f'{tag}_metrics.json').write_text(json.dumps(payload))


class TestIndexRuns:
    def test_pairs_a_private_run_with_its_own_baseline(self, tmp_path):
        _metrics(tmp_path, 'resnet50_centralised_f0_s42_v2')
        _metrics(tmp_path, 'resnet50_dphead_eps2.0_f0_s42_v2')
        index = stats_module.index_runs(stats_module.collect_metrics(str(tmp_path), '*_metrics.json'))
        assert len(index) == 1
        variants = next(iter(index.values()))
        assert 'none' in variants
        assert len(variants) == 2

    def test_a_private_run_never_overwrites_the_baseline(self, tmp_path):
        """The centralised DP script records no dp_mode, so classifying runs by
        that field filed them as non-private and clobbered the baseline."""
        _metrics(tmp_path, 'resnet50_centralised_f0_s42_v2', accuracy=0.80)
        _metrics(tmp_path, 'resnet50_dphead_eps2.0_f0_s42_v2', accuracy=0.28)
        index = stats_module.index_runs(stats_module.collect_metrics(str(tmp_path), '*_metrics.json'))
        assert next(iter(index.values()))['none']['accuracy'] == 0.80

    def test_federated_dp_is_matched_against_federated_not_centralised(self, tmp_path):
        _metrics(tmp_path, 'resnet50_centralised_f0_s42_v2')
        _metrics(tmp_path, 'resnet50_fedavg_K4_T20_E3_f0_s42_v2')
        _metrics(tmp_path, 'resnet50_dpfedavg_userlevel_T20_E3_f0_s42_eps2.0_v2')
        index = stats_module.index_runs(stats_module.collect_metrics(str(tmp_path), '*_metrics.json'))
        federated = index[('resnet50', 'fedavg', 0, 42)]
        assert len(federated) == 2
        assert len(index[('resnet50', 'centralised', 0, 42)]) == 1

    def test_head_and_full_scope_are_separate_conditions(self, tmp_path):
        _metrics(tmp_path, 'resnet50_dpfedavg_userlevel_T20_E3_f0_s42_eps2.0_v2')
        _metrics(tmp_path, 'resnet50_dpfedavg_userlevel_head_T20_E3_f0_s42_eps2.0_v2')
        index = stats_module.index_runs(stats_module.collect_metrics(str(tmp_path), '*_metrics.json'))
        assert len(next(iter(index.values()))) == 2

    def test_a_run_with_no_recognisable_tag_still_indexes(self, tmp_path):
        _metrics(tmp_path, 'mystery')
        index = stats_module.index_runs(stats_module.collect_metrics(str(tmp_path), '*_metrics.json'))
        assert len(index) == 1


class TestBuildComparisons:
    def _index(self, tmp_path, folds):
        for fold in folds:
            _metrics(tmp_path, f'resnet50_centralised_f{fold}_s42_v2', accuracy=0.80)
            _metrics(tmp_path, f'resnet50_dphead_eps2.0_f{fold}_s42_v2', accuracy=0.40)
        return stats_module.index_runs(
            stats_module.collect_metrics(str(tmp_path), '*_metrics.json'))

    def test_reports_accuracy_and_f1_for_each_private_condition(self, tmp_path):
        comparisons = stats_module.build_comparisons(self._index(tmp_path, range(4)))
        assert len(comparisons) == 2
        assert all(c['n_pairs'] == 4 for c in comparisons)

    def test_a_consistent_drop_under_dp_is_significant(self, tmp_path):
        comparisons = stats_module.build_comparisons(self._index(tmp_path, range(6)))
        assert all(c['significant_at_0.05'] for c in comparisons)

    def test_skips_conditions_with_fewer_than_two_matched_pairs(self, tmp_path):
        assert stats_module.build_comparisons(self._index(tmp_path, range(1))) == []

    def test_ignores_a_private_run_whose_split_has_no_baseline(self, tmp_path):
        for fold in range(4):
            _metrics(tmp_path, f'resnet50_centralised_f{fold}_s42_v2', accuracy=0.80)
            _metrics(tmp_path, f'resnet50_dphead_eps2.0_f{fold}_s42_v2', accuracy=0.40)
        # A fold where only the private run exists contributes no pair.
        _metrics(tmp_path, 'resnet50_dphead_eps2.0_f9_s42_v2', accuracy=0.40)
        index = stats_module.index_runs(
            stats_module.collect_metrics(str(tmp_path), '*_metrics.json'))
        assert all(c['n_pairs'] == 4 for c in stats_module.build_comparisons(index))

    def test_no_private_runs_means_no_comparisons(self, tmp_path):
        _metrics(tmp_path, 'resnet50_centralised_f0_s42_v2')
        index = stats_module.index_runs(
            stats_module.collect_metrics(str(tmp_path), '*_metrics.json'))
        assert stats_module.build_comparisons(index) == []


class TestMain:
    def _run(self, results_dir, monkeypatch):
        monkeypatch.setattr('sys.argv', ['stats_analysis.py', '--results-dir', str(results_dir)])
        stats_module.main()
        return json.loads((results_dir / 'metrics' / 'stats_analysis.json').read_text())

    def test_writes_a_report_with_the_chance_band(self, tmp_path, monkeypatch, capsys):
        metrics_dir = tmp_path / 'metrics'
        metrics_dir.mkdir()
        _metrics(metrics_dir, 'resnet50_centralised_f0_s42_v2')
        out = self._run(tmp_path, monkeypatch)
        assert out['n_experiments'] == 1
        assert out['chance_band']['three_class_chance'] == pytest.approx(1 / 3)
        assert 'Found 1 metrics files' in capsys.readouterr().out

    def test_reports_the_paired_comparisons_it_found(self, tmp_path, monkeypatch, capsys):
        metrics_dir = tmp_path / 'metrics'
        metrics_dir.mkdir()
        for fold in range(4):
            _metrics(metrics_dir, f'resnet50_centralised_f{fold}_s42_v2', accuracy=0.80)
            _metrics(metrics_dir, f'resnet50_dphead_eps2.0_f{fold}_s42_v2', accuracy=0.40)
        out = self._run(tmp_path, monkeypatch)
        assert len(out['paired_wilcoxon_dp_vs_nondp']) == 2
        assert 'head-scope' in capsys.readouterr().out

    def test_default_results_dir_points_at_the_projects_own_results(self):
        default = stats_module.default_results_dir()
        assert default.endswith('results')
        assert 'ppxfl-alzheimer' in default.replace(os.sep, '/')
