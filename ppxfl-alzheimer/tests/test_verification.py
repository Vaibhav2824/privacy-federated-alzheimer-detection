"""Validation & Verification against the project's own shipped artefacts.

These tests read the real splits file and the real results JSONs rather than
synthetic fixtures: they verify that what the project actually ships is
internally consistent. They skip when the artefacts are absent so a fresh
clone (no data, no results) still runs green.
"""

import glob
import json
import os

import pytest

import splits as splits_module

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_V2 = os.path.join(PROJECT_ROOT, 'data', 'processed_v2', 'manifest.csv')
SPLITS_V2 = os.path.join(PROJECT_ROOT, 'data', 'splits', 'splits_v2.json')
METRICS_V2 = os.path.join(PROJECT_ROOT, 'results_v2', 'metrics')

REQUIRED_KEYS = ('accuracy', 'f1_macro', 'auroc_macro', 'precision_macro', 'recall_macro')
UNIT_INTERVAL_KEYS = REQUIRED_KEYS
EPSILON_TOLERANCE = 0.15
#: Slack for the accountant's own sigma binary search when checking for overspend.
BUDGET_OVERSHOOT_TOLERANCE = 0.01


def _metrics_files():
    return sorted(glob.glob(os.path.join(METRICS_V2, '*_metrics.json')))


requires_splits = pytest.mark.skipif(
    not (os.path.exists(MANIFEST_V2) and os.path.exists(SPLITS_V2)),
    reason='expanded-cohort manifest/splits not present in this checkout')

requires_metrics = pytest.mark.skipif(
    not _metrics_files(), reason='no results_v2 metrics present in this checkout')


@requires_splits
class TestShippedSplits:
    def test_declares_five_folds(self):
        with open(SPLITS_V2) as handle:
            assert json.load(handle)['k'] == 5

    @pytest.mark.parametrize('fold', range(5))
    def test_fold_partitions_are_subject_disjoint(self, fold):
        split = splits_module.load_split(fold, MANIFEST_V2, SPLITS_V2)
        train, val, test = (set(split[f'{p}_subjects']) for p in ('train', 'val', 'test'))
        assert not train & val
        assert not train & test
        assert not val & test

    @pytest.mark.parametrize('fold', range(5))
    def test_fold_array_indices_never_repeat(self, fold):
        split = splits_module.load_split(fold, MANIFEST_V2, SPLITS_V2)
        combined = [*split['train_idx'], *split['val_idx'], *split['test_idx']]
        assert len(combined) == len(set(combined))

    def test_every_fold_covers_the_whole_cohort(self):
        splits_module.verify_splits(MANIFEST_V2, SPLITS_V2, k=5)


def _load_metrics():
    for path in _metrics_files():
        with open(path) as handle:
            yield os.path.basename(path), json.load(handle)


@requires_metrics
class TestResultsSchema:
    def test_every_run_reports_the_required_metrics(self):
        missing = {name: [k for k in REQUIRED_KEYS if k not in payload]
                   for name, payload in _load_metrics()}
        assert not {k: v for k, v in missing.items() if v}

    def test_every_reported_metric_is_a_valid_proportion(self):
        out_of_range = []
        for name, payload in _load_metrics():
            for key in UNIT_INTERVAL_KEYS:
                value = payload.get(key)
                if value is not None and not 0.0 <= float(value) <= 1.0:
                    out_of_range.append((name, key, value))
        assert not out_of_range

    def test_private_runs_achieve_their_target_epsilon(self):
        """The accountant's epsilon must land near what the run was calibrated for."""
        drifted = []
        for name, payload in _load_metrics():
            target = payload.get('target_epsilon')
            actual = payload.get('actual_epsilon')
            if target in (None, 0) or actual is None:
                continue
            if abs(float(actual) - float(target)) / float(target) > EPSILON_TOLERANCE:
                drifted.append((name, target, actual))
        assert not drifted

    def test_no_private_run_materially_exceeds_its_privacy_budget(self):
        """Opacus calibrates sigma by binary search, so the achieved epsilon can
        land marginally above the target (observed at most ~0.2%). The paper
        reports the achieved epsilon rather than the target for this reason;
        anything beyond BUDGET_OVERSHOOT_TOLERANCE would be a real overspend."""
        overspent = [(name, payload['target_epsilon'], payload['actual_epsilon'])
                     for name, payload in _load_metrics()
                     if payload.get('target_epsilon') and payload.get('actual_epsilon')
                     and float(payload['actual_epsilon'])
                     > float(payload['target_epsilon']) * (1 + BUDGET_OVERSHOOT_TOLERANCE)]
        assert not overspent

    def test_head_scope_runs_record_a_far_smaller_perturbed_dimension(self):
        """Head-scope runs must perturb the classifier head only."""
        head_runs = [(name, payload) for name, payload in _load_metrics()
                     if payload.get('dp_scope') == 'head' and 'perturbed_params' in payload]
        if not head_runs:
            pytest.skip('no head-scope user-level DP runs recorded yet')
        assert all(payload['perturbed_params'] < 100_000 for _, payload in head_runs)
