"""Tests for the paper number-coherence checker.

The checker is only worth having if it actually fails on a stale number, so
most of these tests feed it a paper that is wrong and assert that it says so.
"""

import json

import pytest

import check_paper_numbers as checker

SPLITS = {
    'k': 5,
    'n_subjects': 299,
    'class_counts': {'0': 101, '1': 99, '2': 99},
    'folds': {'0': {
        'train_subjects': [f's{i}' for i in range(233)],
        'val_subjects': [f'v{i}' for i in range(6)],
        'test_subjects': [f't{i}' for i in range(60)],
    }},
}

GOOD_COHORT_PROSE = (
    r'299~unique subjects (99~AD, 99~MCI, 101~CN). Each fold holds out '
    r'60~test subjects and 6~validation subjects, leaving 233~subjects for training.'
)


def summary(*accuracies):
    return {'conditions': [
        {'accuracy_mean': value, 'accuracy_std': None} for value in accuracies
    ]}


class TestAutoBlocks:
    def test_reads_the_body_of_each_block(self):
        text = '% BEGIN AUTO:a\nrow one\n% END AUTO:a\n% BEGIN AUTO:b\n% END AUTO:b'
        blocks = checker.auto_blocks(text)
        assert blocks == {'a': 'row one', 'b': ''}

    def test_a_paper_with_no_blocks_yields_nothing(self):
        assert checker.auto_blocks('plain text') == {}

    def test_refuses_an_unclosed_block(self):
        with pytest.raises(ValueError, match='never closed'):
            checker.auto_blocks('% BEGIN AUTO:a\nrow')


class TestCheckBlocksPopulated:
    def test_passes_when_every_block_has_content(self):
        assert checker.check_blocks_populated('% BEGIN AUTO:a\nrow\n% END AUTO:a') == []

    def test_reports_an_empty_block(self):
        assert checker.check_blocks_populated('% BEGIN AUTO:a\n% END AUTO:a') == ['a']


class TestStripAutoBlocks:
    def test_removes_generated_content(self):
        text = 'prose 44.4\\% % BEGIN AUTO:a\ntable 12.3\\%\n% END AUTO:a'
        assert '12.3' not in checker.strip_auto_blocks(text)
        assert '44.4' in checker.strip_auto_blocks(text)


class TestRecordedPercentages:
    def test_renders_means_the_way_prose_writes_them(self):
        assert '36.1' in checker.recorded_percentages(summary(0.361))

    def test_ignores_a_condition_with_no_accuracy(self):
        assert checker.recorded_percentages(summary(None)) == set()


class TestRawRunPercentages:
    def test_reads_an_individual_run_accuracy(self, tmp_path):
        metrics = tmp_path / 'metrics'
        metrics.mkdir()
        (metrics / 'run_metrics.json').write_text(json.dumps({'accuracy': 0.463}))
        assert '46.3' in checker.raw_run_percentages([str(tmp_path)])

    def test_reads_a_nested_ablation_sweep(self, tmp_path):
        metrics = tmp_path / 'metrics'
        metrics.mkdir()
        (metrics / 'ablation_results.json').write_text(
            json.dumps({'A6_K8': {'accuracy': 0.249}}))
        assert '24.9' in checker.raw_run_percentages([str(tmp_path)])

    def test_skips_a_corrupt_file(self, tmp_path):
        metrics = tmp_path / 'metrics'
        metrics.mkdir()
        (metrics / 'broken.json').write_text('{not json')
        (metrics / 'ok_metrics.json').write_text(json.dumps({'accuracy': 0.5}))
        assert checker.raw_run_percentages([str(tmp_path)]) == {'50.0'}

    def test_missing_directory_yields_nothing(self, tmp_path):
        assert checker.raw_run_percentages([str(tmp_path / 'absent')]) == set()


class TestCheckProsePercentages:
    def test_accepts_a_number_backed_by_a_recorded_result(self):
        assert checker.check_prose_percentages('accuracy was 36.1\\%', summary(0.361)) == []

    def test_flags_a_number_backed_by_nothing(self):
        assert checker.check_prose_percentages('accuracy was 91.1\\%', summary(0.361)) == []
        assert checker.check_prose_percentages('accuracy was 77.7\\%', summary(0.361)) == ['77.7']

    def test_accepts_a_documented_literal(self):
        assert checker.check_prose_percentages('chance is 33.3\\%', summary()) == []

    def test_accepts_a_number_traced_to_a_raw_run(self):
        assert checker.check_prose_percentages(
            'one fold reached 46.3\\%', summary(), raw_values={'46.3'}) == []

    def test_ignores_numbers_inside_generated_blocks(self):
        text = '% BEGIN AUTO:a\n77.7\\%\n% END AUTO:a'
        assert checker.check_prose_percentages(text, summary()) == []


class TestCheckCohortFacts:
    def test_passes_on_prose_that_matches_the_splits_file(self):
        assert checker.check_cohort_facts(GOOD_COHORT_PROSE, SPLITS) == []

    def test_reports_a_missing_cohort_size(self):
        problems = checker.check_cohort_facts('no numbers here', SPLITS)
        assert any('299 subjects' in problem for problem in problems)

    def test_reports_a_wrong_class_count(self):
        prose = GOOD_COHORT_PROSE.replace('99~AD', '95~AD')
        assert any('99~AD' in problem for problem in checker.check_cohort_facts(prose, SPLITS))

    def test_reports_a_missing_fold_size(self):
        prose = GOOD_COHORT_PROSE.replace('60~test subjects', 'some test subjects')
        assert any('test' in problem for problem in checker.check_cohort_facts(prose, SPLITS))


class TestMergeSummaries:
    def test_pools_conditions_from_several_cohorts(self, tmp_path):
        first = tmp_path / 'a.json'
        second = tmp_path / 'b.json'
        first.write_text(json.dumps(summary(0.361)))
        second.write_text(json.dumps(summary(0.926)))
        merged = checker.merge_summaries([str(first), str(second)])
        assert len(merged['conditions']) == 2

    def test_skips_a_cohort_that_is_not_present(self, tmp_path):
        assert checker.merge_summaries([str(tmp_path / 'absent.json')])['conditions'] == []


class TestRunChecks:
    def _project(self, tmp_path, paper_text, accuracies=(0.361,)):
        results = tmp_path / 'results_v2'
        (results / 'metrics').mkdir(parents=True)
        (results / 'metrics' / 'results_summary.json').write_text(
            json.dumps(summary(*accuracies)))
        splits = tmp_path / 'splits.json'
        splits.write_text(json.dumps(SPLITS))
        paper = tmp_path / 'paper.tex'
        paper.write_text(paper_text, encoding='utf-8')
        return str(paper), str(results), str(splits)

    def test_a_coherent_paper_passes_every_check(self, tmp_path):
        text = GOOD_COHORT_PROSE + '\nAccuracy was 36.1\\%.\n% BEGIN AUTO:a\nrow\n% END AUTO:a'
        results = checker.run_checks(*self._project(tmp_path, text))
        assert all(problems == [] for _, problems in results)

    def test_a_stale_number_fails_the_check(self, tmp_path):
        text = GOOD_COHORT_PROSE + '\nAccuracy was 62.5\\%.\n'
        results = dict(checker.run_checks(*self._project(tmp_path, text)))
        assert results['unexplained prose percentages'] == ['62.5']

    def test_an_empty_generated_block_fails_the_check(self, tmp_path):
        text = GOOD_COHORT_PROSE + '\n% BEGIN AUTO:a\n% END AUTO:a'
        results = dict(checker.run_checks(*self._project(tmp_path, text)))
        assert results['empty generated blocks'] == ['a']

    def test_an_extra_cohort_can_explain_a_number(self, tmp_path):
        text = GOOD_COHORT_PROSE + '\nThe earlier cohort reached 92.6\\%.\n'
        paper, results_dir, splits = self._project(tmp_path, text)

        other = tmp_path / 'results'
        (other / 'metrics').mkdir(parents=True)
        (other / 'metrics' / 'results_summary.json').write_text(json.dumps(summary(0.926)))

        without = dict(checker.run_checks(paper, results_dir, splits))
        with_extra = dict(checker.run_checks(paper, results_dir, splits, [str(other)]))
        assert without['unexplained prose percentages'] == []  # 92.6 is an allowed literal
        assert with_extra['unexplained prose percentages'] == []
