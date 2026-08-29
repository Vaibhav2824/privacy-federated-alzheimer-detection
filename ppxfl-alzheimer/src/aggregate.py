"""aggregate.py — Collapse per-run metrics JSONs into per-condition summaries.

Every experiment writes one ``<tag>_metrics.json`` into ``<results>/metrics/``.
The tag encodes the full experimental condition, so aggregation is just: parse
the tags, group the runs that differ only by fold/seed, and report mean +/- std.

This replaces the hardcoded ``aggregate_results.py`` (which enumerated v1 tag
names by hand and could not see the expanded-cohort runs at all). Nothing here
touches the filesystem except :func:`load_records` and :func:`write_summary`,
so the aggregation logic is directly testable.

Usage:
    python -m src.aggregate --results-dir results_v2
"""

import argparse
import glob
import json
import os
import re
import statistics

#: Metrics carried through from each run into the group summary.
METRIC_KEYS = ('accuracy', 'f1_macro', 'auroc_macro', 'precision_macro', 'recall_macro')

#: Chance accuracy for the balanced three-class CN/MCI/AD task.
CHANCE_ACCURACY = 1.0 / 3.0

_FOLD_RE = re.compile(r'_f(\d+)(?:_|$)')
_SEED_RE = re.compile(r'_s(\d+)(?:_|$)')
_EPS_RE = re.compile(r'_eps([0-9.]+?)(?:_|$)')
_CLIENTS_RE = re.compile(r'_K(\d+)(?:_|$)')


def _search_int(pattern, tag):
    match = pattern.search(tag)
    return int(match.group(1)) if match else None


def parse_tag(tag):
    """Decompose an experiment tag into the condition it identifies.

    Returns a dict with ``model``, ``method``, ``dp_scope``, ``epsilon``,
    ``fold``, ``seed``, ``num_clients`` and ``cohort``. ``dp_scope`` and
    ``epsilon`` are ``None`` for non-private runs.
    """
    cohort = 'v2' if tag.endswith('_v2') else 'v1'
    model = tag.split('_', 1)[0]

    eps_match = _EPS_RE.search(tag)
    epsilon = float(eps_match.group(1)) if eps_match else None

    if 'dpfedavg_userlevel' in tag:
        method = 'dpfedavg_userlevel'
        dp_scope = 'head' if 'userlevel_head' in tag else 'full'
    elif 'dphead' in tag:
        method = 'fedavg_dp' if 'fedavg' in tag else 'centralised_dp'
        dp_scope = 'head'
    elif 'dpfull' in tag:
        method = 'fedavg_dp' if 'fedavg' in tag else 'centralised_dp'
        dp_scope = 'full'
    elif 'fedavg' in tag:
        method = 'fedavg'
        dp_scope = None
    else:
        method = 'centralised'
        dp_scope = None

    return {
        'model': model,
        'method': method,
        'dp_scope': dp_scope,
        'epsilon': epsilon,
        'fold': _search_int(_FOLD_RE, tag),
        'seed': _search_int(_SEED_RE, tag),
        'num_clients': _search_int(_CLIENTS_RE, tag),
        'cohort': cohort,
    }


def condition_key(parsed):
    """Group label for a condition: everything except which fold/seed it ran on."""
    parts = [parsed['model'], parsed['method']]
    if parsed['dp_scope']:
        parts.append(f"{parsed['dp_scope']}-scope")
    if parsed['epsilon'] is not None:
        parts.append(f"eps{parsed['epsilon']:g}")
    if parsed['num_clients'] is not None:
        parts.append(f"K{parsed['num_clients']}")
    return ' | '.join(parts)


def mean_std(values):
    """Mean and sample standard deviation, ignoring ``None``. Empty -> (None, None)."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def load_records(metrics_dir, cohort=None):
    """Read every ``*_metrics.json`` in ``metrics_dir`` into parsed records.

    Files that are not valid JSON are skipped rather than aborting the run: a
    partially written metrics file from an interrupted experiment should not
    take down the whole aggregation.
    """
    records = []
    for path in sorted(glob.glob(os.path.join(metrics_dir, '*_metrics.json'))):
        tag = os.path.basename(path)[: -len('_metrics.json')]
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except (ValueError, OSError):
            continue
        parsed = parse_tag(tag)
        if cohort is not None and parsed['cohort'] != cohort:
            continue
        parsed['tag'] = tag
        parsed['metrics'] = {k: payload.get(k) for k in METRIC_KEYS}
        parsed['actual_epsilon'] = payload.get('actual_epsilon')
        parsed['perturbed_params'] = payload.get('perturbed_params')
        records.append(parsed)
    return records


def aggregate(records):
    """Collapse records into one summary per condition, sorted by label."""
    groups = {}
    for record in records:
        groups.setdefault(condition_key(record), []).append(record)

    summaries = []
    for label in sorted(groups):
        members = groups[label]
        summary = {
            'condition': label,
            'model': members[0]['model'],
            'method': members[0]['method'],
            'dp_scope': members[0]['dp_scope'],
            'epsilon': members[0]['epsilon'],
            'n_runs': len(members),
            'folds': sorted({m['fold'] for m in members if m['fold'] is not None}),
            'seeds': sorted({m['seed'] for m in members if m['seed'] is not None}),
            'tags': sorted(m['tag'] for m in members),
        }
        for key in METRIC_KEYS:
            mean, std = mean_std([m['metrics'].get(key) for m in members])
            summary[f'{key}_mean'] = mean
            summary[f'{key}_std'] = std
        actual_eps_mean, _ = mean_std([m['actual_epsilon'] for m in members])
        summary['actual_epsilon_mean'] = actual_eps_mean
        perturbed = {m['perturbed_params'] for m in members if m['perturbed_params'] is not None}
        summary['perturbed_params'] = perturbed.pop() if len(perturbed) == 1 else None
        summaries.append(summary)
    return summaries


def format_table(summaries):
    """Render summaries as fixed-width text rows for the console and the paper."""
    header = f"{'condition':52s} {'n':>2s}  {'accuracy':>14s}  {'f1_macro':>13s}  {'auroc':>13s}"
    lines = [header, '-' * len(header)]
    for s in summaries:
        if s['accuracy_mean'] is None:
            lines.append(f"{s['condition']:52s} {s['n_runs']:2d}  {'no data':>14s}")
            continue
        lines.append(
            f"{s['condition']:52s} {s['n_runs']:2d}  "
            f"{s['accuracy_mean'] * 100:6.1f} +/-{s['accuracy_std'] * 100:5.1f}%  "
            f"{s['f1_macro_mean']:6.3f} +/-{s['f1_macro_std']:5.3f}  "
            f"{s['auroc_macro_mean']:6.3f} +/-{s['auroc_macro_std']:5.3f}"
        )
    return '\n'.join(lines)


def write_summary(summaries, out_path, cohort='v2'):
    """Write the machine-readable summary consumed by the paper and the web UI."""
    payload = {
        'cohort': cohort,
        'chance_accuracy': CHANCE_ACCURACY,
        'n_conditions': len(summaries),
        'conditions': summaries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as handle:
        json.dump(payload, handle, indent=2)
    return payload


def main(argv=None):  # pragma: no cover - thin CLI wrapper over tested functions
    parser = argparse.ArgumentParser(description='Aggregate PPXFL run metrics per condition')
    parser.add_argument('--results-dir', type=str, default='results_v2')
    parser.add_argument('--cohort', type=str, default=None, choices=['v1', 'v2'])
    parser.add_argument('--out', type=str, default=None,
                        help='summary JSON path (default: <results-dir>/metrics/results_summary.json)')
    args = parser.parse_args(argv)

    metrics_dir = os.path.join(args.results_dir, 'metrics')
    records = load_records(metrics_dir, cohort=args.cohort)
    summaries = aggregate(records)
    out_path = args.out or os.path.join(metrics_dir, 'results_summary.json')
    write_summary(summaries, out_path, cohort=args.cohort or 'all')

    print(f"{len(records)} runs -> {len(summaries)} conditions")
    print(format_table(summaries))
    print(f"\nWrote {out_path}")


if __name__ == '__main__':  # pragma: no cover
    main()
