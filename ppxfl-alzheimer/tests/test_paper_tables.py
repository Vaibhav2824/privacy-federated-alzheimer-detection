"""Tests for the generated LaTeX table blocks."""

import pytest

import paper_tables


def condition(method, scope=None, epsilon=None, model='resnet50', **overrides):
    row = {
        'condition': f'{model}/{method}',
        'model': model,
        'method': method,
        'dp_scope': scope,
        'epsilon': epsilon,
        'n_runs': 7,
        'accuracy_mean': 0.361,
        'accuracy_std': 0.072,
        'f1_macro_mean': 0.354,
        'f1_macro_std': 0.072,
        'auroc_macro_mean': 0.579,
        'auroc_macro_std': 0.054,
        'perturbed_params': None,
    }
    row.update(overrides)
    return row


@pytest.fixture
def summary():
    return {
        'cohort': 'v2',
        'chance_accuracy': 1 / 3,
        'n_conditions': 6,
        'conditions': [
            condition('centralised'),
            condition('centralised', model='vgg19', accuracy_mean=0.280, f1_macro_mean=0.252),
            condition('fedavg', accuracy_mean=0.333, f1_macro_mean=0.304),
            condition('centralised_dp', 'head', 2.0, accuracy_mean=0.285),
            condition('dpfedavg_userlevel', 'full', 2.0, n_runs=3, perturbed_params=23508035),
            condition('dpfedavg_userlevel', 'head', 2.0, n_runs=3, perturbed_params=6147),
        ],
    }


class TestFmt:
    def test_renders_a_mean_with_its_spread(self):
        assert paper_tables.fmt(0.361, 0.072, 100, 1) == r'36.1 $\pm$ 7.2'

    def test_renders_a_mean_alone_when_no_spread_is_given(self):
        assert paper_tables.fmt(0.5, None, 100, 1) == '50.0'

    def test_renders_an_en_rule_for_a_missing_value(self):
        assert paper_tables.fmt(None, 0.1) == '---'

    def test_honours_the_requested_precision(self):
        assert paper_tables.fmt(0.354, 0.072, 1, 3) == r'0.354 $\pm$ 0.072'


class TestSelect:
    def test_matches_a_full_specification(self, summary):
        assert paper_tables.select(summary['conditions'], 'centralised')['model'] == 'resnet50'

    def test_distinguishes_the_two_architectures(self, summary):
        vgg = paper_tables.select(summary['conditions'], 'centralised', model='vgg19')
        assert vgg['accuracy_mean'] == 0.280

    def test_distinguishes_the_two_dp_scopes(self, summary):
        full = paper_tables.select(summary['conditions'], 'dpfedavg_userlevel', 'full', 2.0)
        head = paper_tables.select(summary['conditions'], 'dpfedavg_userlevel', 'head', 2.0)
        assert full['perturbed_params'] != head['perturbed_params']

    def test_returns_none_for_a_cell_that_was_never_run(self, summary):
        assert paper_tables.select(summary['conditions'], 'dpfedavg_userlevel', 'head', 99.0) is None


class TestMetricCells:
    def test_produces_four_cells(self, summary):
        assert len(paper_tables.metric_cells(summary['conditions'][0])) == 4

    def test_a_missing_condition_yields_en_rules_and_no_runs(self):
        assert paper_tables.metric_cells(None) == ['---', '---', '---', '0']


class TestCentralisedRows:
    def test_one_row_per_architecture(self, summary):
        rows = paper_tables.centralised_rows(summary).splitlines()
        assert len(rows) == 2
        assert rows[0].startswith('VGG19')
        assert rows[1].startswith('ResNet50')

    def test_every_row_terminates_a_latex_line(self, summary):
        assert all(row.endswith(r'\\') for row in paper_tables.centralised_rows(summary).splitlines())

    def test_omits_an_architecture_that_was_never_run(self):
        data = {'conditions': [condition('centralised')]}
        assert len(paper_tables.centralised_rows(data).splitlines()) == 1


class TestFederatedRows:
    def test_reports_centralised_and_federated(self, summary):
        rows = paper_tables.federated_rows(summary)
        assert 'Centralised ResNet50' in rows
        assert 'FedAvg ResNet50' in rows

    def test_empty_when_neither_was_run(self):
        assert paper_tables.federated_rows({'conditions': []}) == ''


class TestDpSweepRows:
    def test_opens_with_the_non_private_reference(self, summary):
        assert paper_tables.dp_sweep_rows(summary).splitlines()[0].startswith(r'No DP')

    def test_includes_each_budget_that_ran(self, summary):
        assert r'$\varepsilon = 2$' in paper_tables.dp_sweep_rows(summary)

    def test_skips_a_budget_that_was_never_run(self, summary):
        assert r'$\varepsilon = 10$' not in paper_tables.dp_sweep_rows(summary)

    def test_no_baseline_means_no_reference_row(self):
        data = {'conditions': [condition('centralised_dp', 'head', 2.0)]}
        assert 'No DP' not in paper_tables.dp_sweep_rows(data)


class TestUserlevelRows:
    def test_reports_both_scopes(self, summary):
        rows = paper_tables.userlevel_rows(summary)
        assert 'Full model' in rows
        assert 'Head only' in rows

    def test_prints_the_perturbed_dimension_with_latex_thin_spaces(self, summary):
        assert '6{,}147' in paper_tables.userlevel_rows(summary)

    def test_shows_an_en_rule_when_the_dimension_was_not_recorded(self):
        data = {'conditions': [condition('dpfedavg_userlevel', 'head', 2.0)]}
        assert '---' in paper_tables.userlevel_rows(data)

    def test_empty_when_no_subject_level_runs_exist(self):
        assert paper_tables.userlevel_rows({'conditions': []}) == ''

    def test_does_not_end_on_a_dangling_rule(self, summary):
        assert not paper_tables.userlevel_rows(summary).endswith(r'\midrule')


class TestSignificanceRows:
    def test_renders_one_row_per_f1_comparison(self):
        stats = {'paired_wilcoxon_dp_vs_nondp': [
            {'comparison': 'non-DP F1 vs resnet50 | centralised_dp | head-scope | eps2 F1',
             'n_pairs': 7, 'mean_a': 0.354, 'mean_b': 0.258, 'p_value': 0.078,
             'significant_at_0.05': False},
            {'comparison': 'non-DP accuracy vs x accuracy', 'n_pairs': 7, 'mean_a': 0.3,
             'mean_b': 0.2, 'p_value': 0.5, 'significant_at_0.05': False},
        ]}
        rows = paper_tables.significance_rows(stats).splitlines()
        assert len(rows) == 1
        assert rows[0].endswith(r'no \\')

    def test_skips_comparisons_that_could_not_be_computed(self):
        stats = {'paired_wilcoxon_dp_vs_nondp': [
            {'comparison': 'non-DP F1 vs x F1', 'error': 'need >=2 paired matched runs'}]}
        assert paper_tables.significance_rows(stats) == ''

    def test_marks_a_significant_result(self):
        stats = {'paired_wilcoxon_dp_vs_nondp': [
            {'comparison': 'non-DP F1 vs x F1', 'n_pairs': 6, 'mean_a': 0.4, 'mean_b': 0.1,
             'p_value': 0.03, 'significant_at_0.05': True}]}
        assert 'yes' in paper_tables.significance_rows(stats)

    def test_empty_stats_produce_no_rows(self):
        assert paper_tables.significance_rows({}) == ''


class TestCohortFacts:
    def test_defines_the_macros_the_abstract_uses(self, summary):
        facts = paper_tables.cohort_facts(summary)
        for macro in ('ChanceAccuracy', 'TotalRuns', 'NumConditions',
                      'BestNonPrivateAcc', 'BestNonPrivateF'):
            assert macro in facts

    def test_total_runs_sums_every_condition(self, summary):
        # Four seven-run conditions plus two three-run conditions.
        assert r'\newcommand{\TotalRuns}{34}' in paper_tables.cohort_facts(summary)

    def test_best_non_private_ignores_private_conditions(self, summary):
        assert r'{36.1\%}' in paper_tables.cohort_facts(summary)

    def test_omits_the_best_macros_when_no_non_private_run_exists(self):
        data = {'n_conditions': 1, 'conditions': [condition('centralised_dp', 'head', 2.0)]}
        assert 'BestNonPrivateAcc' not in paper_tables.cohort_facts(data)


class TestSplice:
    def test_replaces_the_body_between_the_markers(self):
        text = 'before\n% BEGIN AUTO:x\nold\n% END AUTO:x\nafter'
        assert paper_tables.splice(text, 'x', 'new') == 'before\n% BEGIN AUTO:x\nnew\n% END AUTO:x\nafter'

    def test_leaves_everything_outside_the_markers_alone(self):
        text = 'head\n% BEGIN AUTO:x\nold\n% END AUTO:x\ntail'
        result = paper_tables.splice(text, 'x', 'new')
        assert result.startswith('head') and result.endswith('tail')

    def test_refuses_to_silently_skip_a_missing_block(self):
        with pytest.raises(ValueError, match='no "% BEGIN AUTO:missing"'):
            paper_tables.splice('nothing here', 'missing', 'new')


class TestRender:
    def _paper(self):
        blocks = '\n'.join(
            f'% BEGIN AUTO:{name}\n% END AUTO:{name}' for name in paper_tables.BLOCKS)
        return f'\\documentclass{{article}}\n{blocks}\n'

    def test_fills_every_declared_block(self, summary):
        rendered = paper_tables.render(self._paper(), summary, {})
        assert 'ResNet50' in rendered
        assert r'\newcommand{\TotalRuns}' in rendered

    def test_fills_the_significance_block_when_present(self, summary):
        source = self._paper() + '% BEGIN AUTO:significance\n% END AUTO:significance\n'
        stats = {'paired_wilcoxon_dp_vs_nondp': [
            {'comparison': 'non-DP F1 vs x F1', 'n_pairs': 6, 'mean_a': 0.4, 'mean_b': 0.1,
             'p_value': 0.03, 'significant_at_0.05': True}]}
        assert 'yes' in paper_tables.render(source, summary, stats)

    def test_a_paper_without_a_significance_block_still_renders(self, summary):
        assert paper_tables.render(self._paper(), summary, {})

    def test_rerunning_is_idempotent(self, summary):
        once = paper_tables.render(self._paper(), summary, {})
        assert paper_tables.render(once, summary, {}) == once
