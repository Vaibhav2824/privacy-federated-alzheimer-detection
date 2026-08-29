"""Tests for publication figure rendering.

Figures are checked for the data selection they perform and for producing a
non-empty file. Pixel comparison would be brittle and would not catch the thing
that actually matters here, which is plotting the right numbers.
"""

import json

import pytest

import figures


def condition(method, scope, epsilon, accuracy=0.4, f1=0.3, accuracy_std=0.05, f1_std=0.04):
    return {
        'condition': f'{method}/{scope}/{epsilon}',
        'model': 'resnet50',
        'method': method,
        'dp_scope': scope,
        'epsilon': epsilon,
        'n_runs': 3,
        'accuracy_mean': accuracy,
        'accuracy_std': accuracy_std,
        'f1_macro_mean': f1,
        'f1_macro_std': f1_std,
    }


@pytest.fixture
def summary():
    return {
        'cohort': 'v2',
        'chance_accuracy': 1 / 3,
        'conditions': [
            condition('centralised', None, None, accuracy=0.361, f1=0.354),
            condition('fedavg', None, None, accuracy=0.333, f1=0.304),
            condition('centralised_dp', 'head', 2.0, accuracy=0.285, f1=0.258),
            condition('centralised_dp', 'head', 5.0, accuracy=0.309, f1=0.273),
            condition('centralised_dp', 'head', 10.0, accuracy=0.298, f1=0.271),
            condition('dpfedavg_userlevel', 'full', 2.0, accuracy=0.446, f1=0.251),
            condition('dpfedavg_userlevel', 'head', 2.0, accuracy=0.400, f1=0.290),
            condition('dpfedavg_userlevel', 'head', 5.0, accuracy=0.388, f1=0.332),
        ],
    }


class TestLoadSummary:
    def test_reads_a_summary_from_disk(self, tmp_path, summary):
        path = tmp_path / 'results_summary.json'
        path.write_text(json.dumps(summary))
        assert figures.load_summary(str(path))['cohort'] == 'v2'


class TestFindCondition:
    def test_returns_the_matching_condition(self, summary):
        found = figures.find_condition(summary, 'dpfedavg_userlevel', 'head', 5.0)
        assert found is not None and found['f1_macro_mean'] == 0.332

    def test_returns_none_when_that_cell_was_never_run(self, summary):
        assert figures.find_condition(summary, 'dpfedavg_userlevel', 'head', 10.0) is None

    def test_does_not_confuse_the_two_scopes(self, summary):
        full = figures.find_condition(summary, 'dpfedavg_userlevel', 'full', 2.0)
        head = figures.find_condition(summary, 'dpfedavg_userlevel', 'head', 2.0)
        assert full['accuracy_mean'] != head['accuracy_mean']


class TestSweepSeries:
    def test_returns_the_sweep_ordered_by_budget(self, summary):
        epsilons, means, stds = figures.sweep_series(summary, 'centralised_dp', 'head')
        assert epsilons == [2.0, 5.0, 10.0]
        assert means == [0.285, 0.309, 0.298]
        assert len(stds) == 3

    def test_selects_the_requested_metric(self, summary):
        _, means, _ = figures.sweep_series(summary, 'centralised_dp', 'head', 'f1_macro')
        assert means == [0.258, 0.273, 0.271]

    def test_excludes_non_private_conditions(self, summary):
        epsilons, _, _ = figures.sweep_series(summary, 'centralised', None)
        assert epsilons == []

    def test_drops_a_condition_with_no_recorded_value(self):
        data = {'conditions': [condition('centralised_dp', 'head', 2.0),
                               condition('centralised_dp', 'head', 5.0)]}
        data['conditions'][1]['accuracy_mean'] = None
        epsilons, _, _ = figures.sweep_series(data, 'centralised_dp', 'head')
        assert epsilons == [2.0]

    def test_treats_a_missing_spread_as_zero(self):
        row = condition('centralised_dp', 'head', 2.0)
        row['accuracy_std'] = None
        _, _, stds = figures.sweep_series({'conditions': [row]}, 'centralised_dp', 'head')
        assert stds == [0.0]


class TestBestNonPrivate:
    def test_picks_the_strongest_non_private_mean(self, summary):
        assert figures.best_non_private(summary) == 0.361

    def test_honours_the_requested_metric(self, summary):
        assert figures.best_non_private(summary, 'f1_macro') == 0.354

    def test_returns_none_when_every_condition_is_private(self):
        data = {'conditions': [condition('centralised_dp', 'head', 2.0)]}
        assert figures.best_non_private(data) is None


class TestPlotPrivacyUtility:
    def test_writes_a_figure(self, summary, tmp_path):
        out = tmp_path / 'nested' / 'privacy_utility.png'
        figures.plot_privacy_utility(summary, str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_renders_with_no_non_private_reference(self, tmp_path):
        data = {'chance_accuracy': 1 / 3,
                'conditions': [condition('centralised_dp', 'head', 2.0)]}
        out = tmp_path / 'privacy_utility.png'
        figures.plot_privacy_utility(data, str(out))
        assert out.exists()

    def test_renders_with_no_private_runs_at_all(self, tmp_path):
        data = {'chance_accuracy': 1 / 3,
                'conditions': [condition('centralised', None, None)]}
        out = tmp_path / 'privacy_utility.png'
        figures.plot_privacy_utility(data, str(out))
        assert out.exists()


class TestPlotScopeComparison:
    def test_writes_a_figure_comparing_the_two_scopes(self, summary, tmp_path):
        out = tmp_path / 'scope.png'
        figures.plot_scope_comparison(summary, str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_says_so_when_no_subject_level_runs_exist(self, tmp_path):
        data = {'conditions': [condition('centralised', None, None)]}
        out = tmp_path / 'scope.png'
        figures.plot_scope_comparison(data, str(out))
        assert out.exists()

    def test_renders_when_one_scope_is_missing_a_budget(self, tmp_path):
        data = {'conditions': [
            condition('dpfedavg_userlevel', 'full', 2.0),
            condition('dpfedavg_userlevel', 'head', 5.0),
        ]}
        out = tmp_path / 'scope.png'
        figures.plot_scope_comparison(data, str(out))
        assert out.exists()


def _history(tmp_path, name, rounds=5, percent=False):
    path = tmp_path / name
    accuracy = [40 + i for i in range(rounds)] if percent else [0.4 + i * 0.01 for i in range(rounds)]
    path.write_text(json.dumps({'rounds': list(range(1, rounds + 1)), 'accuracy': accuracy}))
    return str(path)


class TestPlotConvergence:
    def test_plots_each_supplied_history(self, tmp_path):
        histories = [('seed 42', _history(tmp_path, 'a.json')),
                     ('seed 123', _history(tmp_path, 'b.json'))]
        out = tmp_path / 'convergence.png'
        figures.plot_convergence(histories, str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_accepts_accuracy_recorded_as_a_percentage(self, tmp_path):
        out = tmp_path / 'convergence.png'
        figures.plot_convergence([('pct', _history(tmp_path, 'p.json', percent=True))], str(out))
        assert out.exists()

    def test_skips_a_history_file_that_does_not_exist(self, tmp_path):
        out = tmp_path / 'convergence.png'
        figures.plot_convergence([('missing', str(tmp_path / 'absent.json'))], str(out))
        assert out.exists()

    def test_skips_a_run_that_recorded_no_rounds(self, tmp_path):
        empty = tmp_path / 'empty.json'
        empty.write_text(json.dumps({'rounds': [], 'accuracy': []}))
        out = tmp_path / 'convergence.png'
        figures.plot_convergence([('empty', str(empty))], str(out))
        assert out.exists()


class TestPlotCohortComparison:
    def test_writes_a_before_and_after_figure(self, tmp_path):
        out = tmp_path / 'cohort.png'
        figures.plot_cohort_comparison(
            [('32 subjects, slice-level split', 0.926), ('299 subjects, subject-level split', 0.361)],
            str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_renders_more_than_two_bars(self, tmp_path):
        out = tmp_path / 'cohort.png'
        figures.plot_cohort_comparison([('a', 0.9), ('b', 0.5), ('c', 0.36)], str(out))
        assert out.exists()
