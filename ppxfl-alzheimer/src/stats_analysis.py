"""stats_analysis.py — Tier A5: bootstrap CIs, paired significance tests, chance band.

Reads results/metrics/*_metrics.json (+ matching *_run_meta.json for test_subjects)
and the manifest+splits to recompute subject-level bootstrap confidence intervals
on accuracy/F1, plus paired Wilcoxon signed-rank tests across folds for named
experiment-pair comparisons (e.g. non-DP vs DP-eps5, centralised vs FedAvg).

Bootstraps over SUBJECTS (not slices) — resampling slices would understate
variance since a subject's slices are correlated.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import condition_key, parse_tag  # noqa: E402


def subject_level_bootstrap_ci(y_true, y_pred, subject_ids, n_boot=2000, seed=42, alpha=0.05):
    """Bootstrap CI for accuracy and macro-F1 by resampling unique subjects with replacement."""
    rng = np.random.RandomState(seed)
    subjects = np.array(subject_ids)
    unique_subjects = np.unique(subjects)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accs, f1s = [], []
    for _ in range(n_boot):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        np.isin(subjects, sampled)
        # weight by how many times each subject was drawn
        idx = []
        for s in sampled:
            idx.extend(np.where(subjects == s)[0].tolist())
        idx = np.array(idx)
        if len(idx) == 0:  # pragma: no cover - defensive; a non-empty cohort always yields indices
            continue
        acc = float((y_pred[idx] == y_true[idx]).mean())
        from sklearn.metrics import f1_score
        f1 = float(f1_score(y_true[idx], y_pred[idx], average='macro', zero_division=0))
        accs.append(acc)
        f1s.append(f1)

    def ci(vals):
        lo = float(np.percentile(vals, 100 * alpha / 2))
        hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
        return {'mean': float(np.mean(vals)), 'ci_lo': lo, 'ci_hi': hi}

    return {'accuracy': ci(accs), 'f1_macro': ci(f1s), 'n_boot': n_boot}


def paired_wilcoxon(values_a, values_b, label_a, label_b):
    """Paired Wilcoxon signed-rank test across matched folds/seeds."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {'comparison': f'{label_a} vs {label_b}', 'error': 'need >=2 paired matched runs'}
    if np.allclose(a, b):
        return {'comparison': f'{label_a} vs {label_b}', 'statistic': 0.0, 'p_value': 1.0, 'n_pairs': len(a)}
    stat, p = stats.wilcoxon(a, b)
    return {
        'comparison': f'{label_a} vs {label_b}', 'statistic': float(stat), 'p_value': float(p),
        'n_pairs': len(a), 'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
        'significant_at_0.05': bool(p < 0.05),
    }


def collect_metrics(metrics_dir, pattern):
    """Load every metrics JSON matching `pattern`, tagged with its run name."""
    rows = []
    for path in sorted(glob.glob(os.path.join(metrics_dir, pattern))):
        try:
            with open(path) as handle:
                d = json.load(handle)
        except (ValueError, OSError):
            continue
        d['_path'] = path
        d['_tag'] = os.path.basename(path)[: -len('_metrics.json')]
        rows.append(d)
    return rows


#: Which non-private condition each private mechanism should be compared against.
BASELINE_METHOD = {
    'centralised_dp': 'centralised',
    'fedavg_dp': 'fedavg',
    'dpfedavg_userlevel': 'fedavg',
}


def index_runs(records):
    """Index runs by the condition they belong to and the split they ran on.

    Runs are classified from their tag rather than from a `dp_mode` field: the
    centralised DP script records no `dp_mode` at all, so keying on that field
    filed private runs as non-private and let them overwrite the very baselines
    they were meant to be compared against.

    Returns ``{(model, baseline_method, fold, seed): {condition_label: record}}``
    where the non-private run is stored under the key ``'none'``.
    """
    index = {}
    for record in records:
        parsed = parse_tag(record.get('_tag', ''))
        method = parsed['method']
        baseline = BASELINE_METHOD.get(method, method)
        key = (parsed['model'], baseline, parsed['fold'], parsed['seed'])
        label = 'none' if parsed['epsilon'] is None else condition_key(parsed)
        index.setdefault(key, {})[label] = record
    return index


def build_comparisons(index):
    """Paired Wilcoxon of every private condition against its own baseline."""
    labels = sorted({label for variants in index.values() for label in variants if label != 'none'})

    comparisons = []
    for label in labels:
        acc_none, acc_dp, f1_none, f1_dp = [], [], [], []
        for variants in index.values():
            if 'none' in variants and label in variants:
                acc_none.append(variants['none']['accuracy'])
                acc_dp.append(variants[label]['accuracy'])
                f1_none.append(variants['none']['f1_macro'])
                f1_dp.append(variants[label]['f1_macro'])
        if len(acc_none) >= 2:
            comparisons.append(paired_wilcoxon(acc_none, acc_dp, 'non-DP accuracy', f'{label} accuracy'))
            comparisons.append(paired_wilcoxon(f1_none, f1_dp, 'non-DP F1', f'{label} F1'))
    return comparisons


def default_results_dir():
    """The project's own results/ directory, used when --results-dir is omitted."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, 'results')


def main():
    parser = argparse.ArgumentParser(description='PPXFL Tier A5 statistical analysis')
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--n-boot', type=int, default=2000)
    args = parser.parse_args()

    results_dir = args.results_dir or default_results_dir()
    metrics_dir = os.path.join(results_dir, 'metrics')

    all_metrics = collect_metrics(metrics_dir, '*_metrics.json')
    print(f"Found {len(all_metrics)} metrics files")

    comparisons = build_comparisons(index_runs(all_metrics))

    out = {
        'n_experiments': len(all_metrics),
        'paired_wilcoxon_dp_vs_nondp': comparisons,
        'chance_band': {'three_class_chance': 1.0 / 3.0},
    }
    out_path = os.path.join(metrics_dir, 'stats_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    for c in comparisons:
        print(f"  {c.get('comparison')}: p={c.get('p_value')}, "
              f"sig={c.get('significant_at_0.05')}, n={c.get('n_pairs')}")


if __name__ == '__main__':
    main()
