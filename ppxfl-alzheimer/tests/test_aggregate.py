"""Tests for tag parsing and per-condition aggregation of run metrics."""

import json

import pytest

import aggregate as agg


class TestParseTag:
    def test_centralised_tag(self):
        parsed = agg.parse_tag('resnet50_centralised_f2_s123_v2')
        assert parsed['model'] == 'resnet50'
        assert parsed['method'] == 'centralised'
        assert parsed['dp_scope'] is None
        assert parsed['epsilon'] is None
        assert parsed['fold'] == 2
        assert parsed['seed'] == 123
        assert parsed['cohort'] == 'v2'

    def test_v1_tags_are_recognised_by_the_missing_suffix(self):
        assert agg.parse_tag('resnet50_centralised_f0_s42')['cohort'] == 'v1'

    def test_fedavg_tag_records_the_client_count(self):
        parsed = agg.parse_tag('resnet50_fedavg_K4_T20_E3_f0_s42_v2')
        assert parsed['method'] == 'fedavg'
        assert parsed['num_clients'] == 4
        assert parsed['dp_scope'] is None

    def test_centralised_dp_head_tag(self):
        parsed = agg.parse_tag('resnet50_dphead_eps2.0_f1_s42_v2')
        assert parsed['method'] == 'centralised_dp'
        assert parsed['dp_scope'] == 'head'
        assert parsed['epsilon'] == 2.0

    def test_centralised_dp_full_tag(self):
        parsed = agg.parse_tag('resnet50_dpfull_eps5.0_f0_s42_v2')
        assert parsed['method'] == 'centralised_dp'
        assert parsed['dp_scope'] == 'full'

    def test_dp_federated_tag_is_distinguished_from_centralised_dp(self):
        parsed = agg.parse_tag('resnet50_fedavg_K4_T20_E3_f0_s42_dphead_eps5.0')
        assert parsed['method'] == 'fedavg_dp'
        assert parsed['dp_scope'] == 'head'

    def test_dp_federated_full_scope_tag(self):
        parsed = agg.parse_tag('resnet50_fedavg_K4_T20_E3_f0_s42_dpfull_eps5.0')
        assert parsed['method'] == 'fedavg_dp'
        assert parsed['dp_scope'] == 'full'

    def test_userlevel_full_scope_tag(self):
        parsed = agg.parse_tag('resnet50_dpfedavg_userlevel_T20_E3_f0_s42_eps2.0_v2')
        assert parsed['method'] == 'dpfedavg_userlevel'
        assert parsed['dp_scope'] == 'full'
        assert parsed['epsilon'] == 2.0

    def test_userlevel_head_scope_tag(self):
        parsed = agg.parse_tag('resnet50_dpfedavg_userlevel_head_T20_E3_f0_s2024_eps10.0_v2')
        assert parsed['method'] == 'dpfedavg_userlevel'
        assert parsed['dp_scope'] == 'head'
        assert parsed['epsilon'] == 10.0
        assert parsed['seed'] == 2024

    def test_missing_fold_and_seed_are_none(self):
        parsed = agg.parse_tag('resnet50_centralised')
        assert parsed['fold'] is None
        assert parsed['seed'] is None


class TestConditionKey:
    def test_runs_differing_only_by_fold_and_seed_share_a_key(self):
        a = agg.parse_tag('resnet50_dphead_eps2.0_f0_s42_v2')
        b = agg.parse_tag('resnet50_dphead_eps2.0_f4_s2024_v2')
        assert agg.condition_key(a) == agg.condition_key(b)

    def test_different_scopes_get_different_keys(self):
        full = agg.parse_tag('resnet50_dpfedavg_userlevel_T20_E3_f0_s42_eps2.0_v2')
        head = agg.parse_tag('resnet50_dpfedavg_userlevel_head_T20_E3_f0_s42_eps2.0_v2')
        assert agg.condition_key(full) != agg.condition_key(head)

    def test_key_includes_client_count_when_present(self):
        assert 'K4' in agg.condition_key(agg.parse_tag('resnet50_fedavg_K4_T20_E3_f0_s42_v2'))

    def test_non_private_key_omits_scope_and_epsilon(self):
        assert agg.condition_key(agg.parse_tag('vgg19_centralised_f0_s42_v2')) == 'vgg19 | centralised'


class TestMeanStd:
    def test_mean_and_sample_std_of_several_values(self):
        mean, std = agg.mean_std([1.0, 2.0, 3.0])
        assert mean == pytest.approx(2.0)
        assert std == pytest.approx(1.0)

    def test_single_value_has_zero_spread(self):
        assert agg.mean_std([0.5]) == (0.5, 0.0)

    def test_none_entries_are_dropped(self):
        assert agg.mean_std([None, 4.0, None]) == (4.0, 0.0)

    def test_all_missing_returns_none(self):
        assert agg.mean_std([None, None]) == (None, None)

    def test_empty_returns_none(self):
        assert agg.mean_std([]) == (None, None)


def _write_metrics(directory, tag, **values):
    payload = {'accuracy': 0.5, 'f1_macro': 0.4, 'auroc_macro': 0.6,
               'precision_macro': 0.4, 'recall_macro': 0.4}
    payload.update(values)
    path = directory / f'{tag}_metrics.json'
    path.write_text(json.dumps(payload))
    return path


class TestLoadRecords:
    def test_reads_every_metrics_file(self, tmp_path):
        _write_metrics(tmp_path, 'resnet50_centralised_f0_s42_v2')
        _write_metrics(tmp_path, 'resnet50_centralised_f1_s42_v2')
        assert len(agg.load_records(str(tmp_path))) == 2

    def test_ignores_non_metrics_files(self, tmp_path):
        _write_metrics(tmp_path, 'resnet50_centralised_f0_s42_v2')
        (tmp_path / 'resnet50_centralised_f0_s42_v2_history.json').write_text('{}')
        records = agg.load_records(str(tmp_path))
        assert [r['tag'] for r in records] == ['resnet50_centralised_f0_s42_v2']

    def test_skips_corrupt_json_instead_of_aborting(self, tmp_path):
        _write_metrics(tmp_path, 'resnet50_centralised_f0_s42_v2')
        (tmp_path / 'broken_metrics.json').write_text('{not json')
        assert len(agg.load_records(str(tmp_path))) == 1

    def test_cohort_filter_selects_one_cohort(self, tmp_path):
        _write_metrics(tmp_path, 'resnet50_centralised_f0_s42_v2')
        _write_metrics(tmp_path, 'resnet50_centralised_f0_s42')
        assert len(agg.load_records(str(tmp_path), cohort='v2')) == 1
        assert len(agg.load_records(str(tmp_path), cohort='v1')) == 1

    def test_carries_through_epsilon_and_perturbed_params(self, tmp_path):
        _write_metrics(tmp_path, 'resnet50_dpfedavg_userlevel_head_T20_E3_f0_s42_eps2.0_v2',
                       actual_epsilon=1.99, perturbed_params=6147)
        record = agg.load_records(str(tmp_path))[0]
        assert record['actual_epsilon'] == 1.99
        assert record['perturbed_params'] == 6147

    def test_missing_directory_yields_no_records(self, tmp_path):
        assert agg.load_records(str(tmp_path / 'absent')) == []


class TestAggregate:
    def test_groups_runs_of_one_condition(self, tmp_path):
        _write_metrics(tmp_path, 'resnet50_centralised_f0_s42_v2', accuracy=0.4)
        _write_metrics(tmp_path, 'resnet50_centralised_f1_s42_v2', accuracy=0.6)
        summary = agg.aggregate(agg.load_records(str(tmp_path)))[0]
        assert summary['n_runs'] == 2
        assert summary['accuracy_mean'] == pytest.approx(0.5)
        assert summary['folds'] == [0, 1]
        assert summary['seeds'] == [42]

    def test_separates_distinct_conditions_and_sorts_them(self, tmp_path):
        _write_metrics(tmp_path, 'vgg19_centralised_f0_s42_v2')
        _write_metrics(tmp_path, 'resnet50_centralised_f0_s42_v2')
        labels = [s['condition'] for s in agg.aggregate(agg.load_records(str(tmp_path)))]
        assert labels == sorted(labels)
        assert len(labels) == 2

    def test_reports_perturbed_params_when_the_group_agrees(self, tmp_path):
        for seed in (42, 123):
            _write_metrics(tmp_path, f'resnet50_dpfedavg_userlevel_head_T20_E3_f0_s{seed}_eps2.0_v2',
                           perturbed_params=6147)
        assert agg.aggregate(agg.load_records(str(tmp_path)))[0]['perturbed_params'] == 6147

    def test_suppresses_perturbed_params_when_the_group_disagrees(self, tmp_path):
        _write_metrics(tmp_path, 'resnet50_dpfedavg_userlevel_head_T20_E3_f0_s42_eps2.0_v2',
                       perturbed_params=6147)
        _write_metrics(tmp_path, 'resnet50_dpfedavg_userlevel_head_T20_E3_f0_s123_eps2.0_v2',
                       perturbed_params=99)
        assert agg.aggregate(agg.load_records(str(tmp_path)))[0]['perturbed_params'] is None

    def test_no_records_gives_no_summaries(self):
        assert agg.aggregate([]) == []


class TestFormatTable:
    def test_renders_one_row_per_condition(self, tmp_path):
        _write_metrics(tmp_path, 'resnet50_centralised_f0_s42_v2')
        table = agg.format_table(agg.aggregate(agg.load_records(str(tmp_path))))
        assert 'resnet50 | centralised' in table
        assert len(table.splitlines()) == 3  # header, rule, one row

    def test_marks_conditions_with_no_usable_metrics(self, tmp_path):
        _write_metrics(tmp_path, 'resnet50_centralised_f0_s42_v2', accuracy=None,
                       f1_macro=None, auroc_macro=None)
        assert 'no data' in agg.format_table(agg.aggregate(agg.load_records(str(tmp_path))))


class TestWriteSummary:
    def test_writes_the_summary_the_ui_and_paper_consume(self, tmp_path):
        _write_metrics(tmp_path, 'resnet50_centralised_f0_s42_v2')
        summaries = agg.aggregate(agg.load_records(str(tmp_path)))
        out_path = tmp_path / 'nested' / 'results_summary.json'
        payload = agg.write_summary(summaries, str(out_path))
        written = json.loads(out_path.read_text())
        assert written == payload
        assert written['n_conditions'] == 1
        assert written['chance_accuracy'] == pytest.approx(1 / 3)
        assert written['cohort'] == 'v2'
