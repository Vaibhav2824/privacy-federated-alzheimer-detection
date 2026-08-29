"""Tests for Dirichlet non-IID partitioning across simulated hospital clients."""

import json
import os

import numpy as np
import pytest

import partition


def _subjects(n_per_class=8, n_classes=3):
    ids, labels = [], []
    for label in range(n_classes):
        for i in range(n_per_class):
            ids.append(f'S{label}{i:02d}')
            labels.append(label)
    return np.array(ids), np.array(labels)


class TestDirichletPartitionSubjects:
    def test_every_subject_lands_on_exactly_one_client(self):
        ids, labels = _subjects()
        assigned = partition.dirichlet_partition_subjects(ids, labels, num_clients=4, alpha=0.5)
        flat = [s for subjects in assigned.values() for s in subjects]
        assert sorted(flat) == sorted(ids.tolist())
        assert len(flat) == len(set(flat))

    def test_produces_the_requested_number_of_clients(self):
        ids, labels = _subjects()
        assigned = partition.dirichlet_partition_subjects(ids, labels, num_clients=6, alpha=0.5)
        assert sorted(assigned) == list(range(6))

    def test_no_client_is_left_empty(self):
        ids, labels = _subjects()
        for alpha in (0.1, 0.5, 100.0):
            assigned = partition.dirichlet_partition_subjects(ids, labels, num_clients=4, alpha=alpha)
            assert all(len(v) > 0 for v in assigned.values())

    def test_same_seed_reproduces_the_same_partition(self):
        ids, labels = _subjects()
        a = partition.dirichlet_partition_subjects(ids, labels, num_clients=4, seed=3)
        b = partition.dirichlet_partition_subjects(ids, labels, num_clients=4, seed=3)
        assert a == b

    def test_different_seeds_give_different_partitions(self):
        ids, labels = _subjects()
        a = partition.dirichlet_partition_subjects(ids, labels, num_clients=4, seed=1)
        b = partition.dirichlet_partition_subjects(ids, labels, num_clients=4, seed=2)
        assert a != b

    def test_large_alpha_spreads_subjects_more_evenly_than_small_alpha(self):
        ids, labels = _subjects(n_per_class=20)

        def imbalance(alpha):
            assigned = partition.dirichlet_partition_subjects(ids, labels, num_clients=4,
                                                              alpha=alpha, seed=0)
            sizes = [len(v) for v in assigned.values()]
            return max(sizes) - min(sizes)

        assert imbalance(100.0) < imbalance(0.05)

    def test_rejects_more_clients_than_subjects(self):
        ids, labels = _subjects(n_per_class=1)
        with pytest.raises(ValueError, match='fewer subjects than clients'):
            partition.dirichlet_partition_subjects(ids, labels, num_clients=10)

    def test_gives_up_loudly_when_no_non_empty_partition_is_found(self, monkeypatch):
        """Rather than emit a partition that crashes a later training script."""

        class DegenerateRandomState(np.random.RandomState):
            """Always hands every subject of a class to client 0."""

            def dirichlet(self, alphas, size=None):
                return np.array([1.0] + [0.0] * (len(alphas) - 1))

        ids, labels = _subjects(n_per_class=2)
        monkeypatch.setattr(partition.np.random, 'RandomState', DegenerateRandomState)
        with pytest.raises(RuntimeError, match='every client'):
            partition.dirichlet_partition_subjects(ids, labels, num_clients=5, alpha=0.5)


class TestExpandSubjectsToIndices:
    def test_maps_each_client_to_its_subjects_slices(self, tiny_manifest):
        client_subjects = {0: ['S000', 'S001'], 1: ['S100']}
        indices = partition.expand_subjects_to_indices(client_subjects, tiny_manifest)
        assert len(indices[0]) == 8
        assert len(indices[1]) == 4

    def test_clients_never_share_a_slice(self, tiny_manifest):
        client_subjects = {0: ['S000'], 1: ['S001']}
        indices = partition.expand_subjects_to_indices(client_subjects, tiny_manifest)
        assert not set(indices[0]) & set(indices[1])

    def test_an_empty_client_maps_to_no_slices(self, tiny_manifest):
        assert len(partition.expand_subjects_to_indices({0: []}, tiny_manifest)[0]) == 0


class TestCreateClientDatasets:
    def test_writes_per_client_arrays_and_class_folders(self, tiny_arrays, tmp_path):
        images, labels = tiny_arrays
        client_indices = {0: np.arange(0, 8), 1: np.arange(20, 28)}
        partition.create_client_datasets(images, labels, client_indices, str(tmp_path))
        for client in (1, 2):
            client_dir = tmp_path / f'client_{client}'
            assert (client_dir / 'images.npy').exists()
            assert np.load(client_dir / 'images.npy').shape == (8, 8, 8)
            assert set(os.listdir(client_dir)) >= {'CN', 'MCI', 'AD'}

    def test_saves_one_npy_per_slice_under_its_class(self, tiny_arrays, tmp_path):
        images, labels = tiny_arrays
        partition.create_client_datasets(images, labels, {0: np.arange(0, 4)}, str(tmp_path))
        assert len(os.listdir(tmp_path / 'client_1' / 'CN')) == 4
        assert os.listdir(tmp_path / 'client_1' / 'MCI') == []


class TestComputePartitionStats:
    def test_counts_each_clients_slices_by_class(self, tiny_arrays):
        _, labels = tiny_arrays
        stats = partition.compute_partition_stats(labels, {0: np.arange(0, 20), 1: np.arange(20, 40)})
        assert stats['client_1']['CN'] == 20
        assert stats['client_2']['MCI'] == 20
        assert stats['client_1']['total'] == 20

    def test_percentages_are_over_the_partitioned_pool_not_the_whole_dataset(self, tiny_arrays):
        _, labels = tiny_arrays
        stats = partition.compute_partition_stats(labels, {0: np.arange(0, 10), 1: np.arange(10, 20)})
        assert stats['client_1']['percentage'] == 50.0
        assert stats['client_2']['percentage'] == 50.0

    def test_accepts_a_custom_class_naming(self, tiny_arrays):
        _, labels = tiny_arrays
        stats = partition.compute_partition_stats(labels, {0: np.arange(0, 20)},
                                                  class_names={0: 'A', 1: 'B', 2: 'C'})
        assert stats['client_1']['A'] == 20


class TestPlotPartitionDistribution:
    def test_writes_a_figure_to_the_requested_path(self, tiny_arrays, tmp_path):
        _, labels = tiny_arrays
        out = tmp_path / 'partition.png'
        partition.plot_partition_distribution(labels, {0: np.arange(0, 20), 1: np.arange(20, 40)},
                                              save_path=str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_runs_without_a_save_path(self, tiny_arrays):
        _, labels = tiny_arrays
        partition.plot_partition_distribution(labels, {0: np.arange(0, 20)})


class TestMain:
    """End-to-end CLI run over a synthetic cohort, writing into tmp_path only."""

    @pytest.fixture
    def processed_dir(self, tiny_manifest, tiny_arrays, tmp_path):
        images, labels = tiny_arrays
        directory = tmp_path / 'processed'
        directory.mkdir()
        np.save(directory / 'all_images.npy', images)
        np.save(directory / 'all_labels.npy', labels)
        tiny_manifest.to_csv(directory / 'manifest.csv', index=False)
        return directory

    @pytest.fixture
    def splits_path(self, processed_dir, tmp_path):
        import splits as splits_module
        path = tmp_path / 'splits.json'
        splits_module.build_splits(str(processed_dir / 'manifest.csv'), str(path), k=3, seed=42)
        return path

    def _argv(self, processed_dir, splits_path, tmp_path, extra=()):
        return ['partition.py',
                '--processed-dir', str(processed_dir),
                '--splits', str(splits_path),
                '--fold', '0',
                '--output-root', str(tmp_path / 'clients'),
                '--figures-dir', str(tmp_path / 'figures'),
                '--num-clients', '2',
                '--alpha', '0.5',
                '--seed', '42', *extra]

    def test_writes_client_datasets_metadata_and_figure(self, processed_dir, splits_path,
                                                        tmp_path, monkeypatch):
        monkeypatch.setattr('sys.argv', self._argv(processed_dir, splits_path, tmp_path))
        partition.main()
        out_dir = tmp_path / 'clients' / 'f0_a0.5_s42'
        metadata = json.loads((out_dir / 'partition_metadata.json').read_text())
        assert metadata['num_clients'] == 2
        assert metadata['fold'] == 0
        assert sorted(metadata['client_subjects']) == ['0', '1']
        assert (out_dir / 'partition_indices.json').exists()
        assert (out_dir / 'client_1' / 'images.npy').exists()
        assert (tmp_path / 'figures' / 'partition_f0_alpha0.5.png').exists()

    def test_clients_never_receive_held_out_subjects(self, processed_dir, splits_path,
                                                     tmp_path, monkeypatch):
        import splits as splits_module
        monkeypatch.setattr('sys.argv', self._argv(processed_dir, splits_path, tmp_path))
        partition.main()
        metadata = json.loads(
            (tmp_path / 'clients' / 'f0_a0.5_s42' / 'partition_metadata.json').read_text())
        split = splits_module.load_split(0, str(processed_dir / 'manifest.csv'), str(splits_path))
        holdout = set(split['val_subjects']) | set(split['test_subjects'])
        for subjects in metadata['client_subjects'].values():
            assert not set(subjects) & holdout

    def test_exits_when_the_preprocessed_data_is_missing(self, splits_path, tmp_path, monkeypatch):
        monkeypatch.setattr('sys.argv', self._argv(tmp_path / 'absent', splits_path, tmp_path))
        with pytest.raises(SystemExit) as exit_info:
            partition.main()
        assert exit_info.value.code == 1

    def test_falls_back_to_the_project_default_paths(self, monkeypatch, capsys):
        """With no paths given the CLI resolves the project's own data/ layout."""
        monkeypatch.setattr('sys.argv', ['partition.py', '--fold', '0'])
        monkeypatch.setattr(partition.os.path, 'exists', lambda path: False)
        with pytest.raises(SystemExit):
            partition.main()
        assert 'data' in capsys.readouterr().out
