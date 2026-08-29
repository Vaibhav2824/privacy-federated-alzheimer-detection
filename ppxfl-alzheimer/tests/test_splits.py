"""Tests for subject-level cross-validation splits.

The property these tests exist to protect: no subject may appear in more than
one partition of a fold. Splitting at the slice level instead inflates every
downstream metric, which is exactly the leakage this project set out to remove.
"""

import json

import pandas as pd
import pytest

import splits as splits_module


class TestSubjectTable:
    def test_collapses_slices_to_one_row_per_subject(self, tiny_manifest):
        table = splits_module._subject_table(tiny_manifest)
        assert len(table) == tiny_manifest['subject_id'].nunique()
        assert list(table.columns) == ['subject_id', 'label']

    def test_rejects_a_subject_with_conflicting_labels(self, tiny_manifest):
        corrupted = tiny_manifest.copy()
        corrupted.loc[0, 'label'] = 2
        with pytest.raises(ValueError, match='inconsistent class labels'):
            splits_module._subject_table(corrupted)


class TestBuildSplits:
    def test_writes_all_folds_with_the_requested_k(self, tiny_manifest_path, tmp_path):
        out = str(tmp_path / 'splits' / 's.json')
        result = splits_module.build_splits(tiny_manifest_path, out, k=3, seed=42)
        assert result['k'] == 3
        assert set(result['folds']) == {'0', '1', '2'}
        assert json.loads(open(out).read())['n_subjects'] == 15

    def test_partitions_never_share_a_subject(self, tiny_manifest_path, tmp_path):
        result = splits_module.build_splits(tiny_manifest_path, str(tmp_path / 's.json'), k=3)
        for fold in result['folds'].values():
            train, val, test = (set(fold[f'{p}_subjects']) for p in ('train', 'val', 'test'))
            assert not train & val
            assert not train & test
            assert not val & test

    def test_every_fold_covers_the_whole_cohort(self, tiny_manifest_path, tmp_path):
        result = splits_module.build_splits(tiny_manifest_path, str(tmp_path / 's.json'), k=3)
        for fold in result['folds'].values():
            covered = set(fold['train_subjects']) | set(fold['val_subjects']) | set(fold['test_subjects'])
            assert len(covered) == 15

    def test_same_seed_reproduces_the_same_split(self, tiny_manifest_path, tmp_path):
        a = splits_module.build_splits(tiny_manifest_path, str(tmp_path / 'a.json'), k=3, seed=7)
        b = splits_module.build_splits(tiny_manifest_path, str(tmp_path / 'b.json'), k=3, seed=7)
        assert a['folds'] == b['folds']

    def test_records_the_class_distribution(self, tiny_manifest_path, tmp_path):
        result = splits_module.build_splits(tiny_manifest_path, str(tmp_path / 's.json'), k=3)
        assert sum(result['class_counts'].values()) == 15

    def test_rejects_more_folds_than_subjects(self, tmp_path):
        manifest = pd.DataFrame([
            {'subject_id': f'S{i}', 'label': i % 3, 'array_index': i} for i in range(3)
        ])
        path = tmp_path / 'small.csv'
        manifest.to_csv(path, index=False)
        with pytest.raises(ValueError, match='reduce k'):
            splits_module.build_splits(str(path), str(tmp_path / 's.json'), k=5)


class TestLoadSplit:
    @pytest.fixture
    def built(self, tiny_manifest_path, tmp_path):
        path = str(tmp_path / 's.json')
        splits_module.build_splits(tiny_manifest_path, path, k=3, seed=42)
        return tiny_manifest_path, path

    def test_returns_array_indices_for_each_partition(self, built):
        manifest_path, splits_path = built
        split = splits_module.load_split(0, manifest_path, splits_path)
        assert split['fold'] == 0
        assert len(split['train_idx']) + len(split['val_idx']) + len(split['test_idx']) == 60

    def test_array_indices_never_repeat_across_partitions(self, built):
        manifest_path, splits_path = built
        split = splits_module.load_split(1, manifest_path, splits_path)
        combined = list(split['train_idx']) + list(split['val_idx']) + list(split['test_idx'])
        assert len(combined) == len(set(combined))

    def test_every_slice_of_a_subject_lands_in_one_partition(self, built, tiny_manifest):
        manifest_path, splits_path = built
        split = splits_module.load_split(0, manifest_path, splits_path)
        train_subjects = set(split['train_subjects'])
        expected = tiny_manifest[tiny_manifest['subject_id'].isin(train_subjects)]['array_index']
        assert sorted(split['train_idx']) == sorted(expected.tolist())


class TestVerifySplits:
    def test_passes_on_a_freshly_built_split(self, tiny_manifest_path, tmp_path, capsys):
        path = str(tmp_path / 's.json')
        splits_module.build_splits(tiny_manifest_path, path, k=3, seed=42)
        splits_module.verify_splits(tiny_manifest_path, path, k=3)
        assert 'all 3 folds OK' in capsys.readouterr().out

    def test_load_split_rejects_a_subject_listed_in_two_partitions(self, tiny_manifest_path, tmp_path):
        """The array-index check is the first gate leakage hits."""
        path = tmp_path / 's.json'
        splits_module.build_splits(tiny_manifest_path, str(path), k=3, seed=42)
        payload = json.loads(path.read_text())
        payload['folds']['0']['train_subjects'].append(payload['folds']['0']['test_subjects'][0])
        path.write_text(json.dumps(payload))
        with pytest.raises(AssertionError, match='duplicate array indices'):
            splits_module.verify_splits(tiny_manifest_path, str(path), k=3)

    @pytest.mark.parametrize('partitions, message', [
        (('train', 'val'), 'train/val overlap'),
        (('train', 'test'), 'train/test overlap'),
        (('val', 'test'), 'val/test overlap'),
    ])
    def test_reports_which_partitions_a_shared_subject_crosses(self, tiny_manifest_path, tmp_path,
                                                               partitions, message):
        """A subject id with no manifest rows isolates verify_splits' own overlap
        checks from the earlier duplicate-array-index assertion in load_split."""
        path = tmp_path / 's.json'
        splits_module.build_splits(tiny_manifest_path, str(path), k=3, seed=42)
        payload = json.loads(path.read_text())
        for partition in partitions:
            payload['folds']['0'][f'{partition}_subjects'].append('GHOST')
        path.write_text(json.dumps(payload))
        with pytest.raises(AssertionError, match=message):
            splits_module.verify_splits(tiny_manifest_path, str(path), k=3)

    def test_fails_when_a_subject_is_missing_from_every_partition(self, tiny_manifest_path, tmp_path):
        path = tmp_path / 's.json'
        splits_module.build_splits(tiny_manifest_path, str(path), k=3, seed=42)
        payload = json.loads(path.read_text())
        payload['folds']['0']['train_subjects'].pop()
        path.write_text(json.dumps(payload))
        with pytest.raises(AssertionError, match='subjects missing'):
            splits_module.verify_splits(tiny_manifest_path, str(path), k=3)


class TestCli:
    def test_build_mode_writes_and_verifies(self, tiny_manifest_path, tmp_path, monkeypatch, capsys):
        out = str(tmp_path / 's.json')
        monkeypatch.setattr('sys.argv', ['splits.py', '--manifest', tiny_manifest_path,
                                         '--out', out, '--k', '3', '--seed', '42'])
        splits_module.main()
        captured = capsys.readouterr().out
        assert 'Built 3-fold subject-level splits' in captured
        assert 'all 3 folds OK' in captured

    def test_verify_mode_checks_an_existing_file(self, tiny_manifest_path, tmp_path,
                                                 monkeypatch, capsys):
        out = str(tmp_path / 's.json')
        splits_module.build_splits(tiny_manifest_path, out, k=3, seed=42)
        monkeypatch.setattr('sys.argv', ['splits.py', '--manifest', tiny_manifest_path,
                                         '--out', out, '--k', '3', '--verify'])
        splits_module.main()
        assert 'all 3 folds OK' in capsys.readouterr().out
