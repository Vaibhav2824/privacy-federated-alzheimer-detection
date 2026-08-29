"""figures.py — Publication figures built from the aggregated results.

Every figure reads `results_summary.json` (written by `src/aggregate.py`) or the
per-run history files, so a figure can never disagree with the tables in the
paper or with the web dashboard. Nothing here re-derives a metric.

Usage:
    python -m src.aggregate --results-dir results_v2 --cohort v2
    python -m src.figures --results-dir results_v2 --out-dir ../figures
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import CHANCE_ACCURACY  # noqa: E402

DPI = 300

#: Colour-blind-safe qualitative colours (Okabe-Ito).
COLOURS = {
    'centralised_dp_head': '#0072B2',
    'dpfedavg_userlevel_full': '#D55E00',
    'dpfedavg_userlevel_head': '#009E73',
    'reference': '#666666',
}

MECHANISMS = [
    ('centralised_dp', 'head', 'Centralised, sample-level DP (head)'),
    ('dpfedavg_userlevel', 'full', 'Subject-level DP-FedAvg (full model)'),
    ('dpfedavg_userlevel', 'head', 'Subject-level DP-FedAvg (head only)'),
]


def load_summary(path):
    """Read a results_summary.json written by src.aggregate."""
    with open(path) as handle:
        return json.load(handle)


def find_condition(summary, method, scope, epsilon):
    """The one summarised condition matching a mechanism, scope and budget."""
    for condition in summary['conditions']:
        if (condition['method'] == method
                and condition['dp_scope'] == scope
                and condition['epsilon'] == epsilon):
            return condition
    return None


def sweep_series(summary, method, scope, metric='accuracy'):
    """``(epsilons, means, stds)`` for one mechanism, ordered by budget.

    Conditions with no recorded value for the metric are dropped rather than
    plotted at zero, which would invent a data point.
    """
    rows = [
        condition for condition in summary['conditions']
        if condition['method'] == method
        and condition['dp_scope'] == scope
        and condition['epsilon'] is not None
        and condition.get(f'{metric}_mean') is not None
    ]
    rows.sort(key=lambda condition: condition['epsilon'])
    return (
        [row['epsilon'] for row in rows],
        [row[f'{metric}_mean'] for row in rows],
        [row.get(f'{metric}_std') or 0.0 for row in rows],
    )


def best_non_private(summary, metric='accuracy'):
    """The strongest non-private condition's mean, used as the utility ceiling."""
    values = [
        condition[f'{metric}_mean'] for condition in summary['conditions']
        if condition['epsilon'] is None and condition.get(f'{metric}_mean') is not None
    ]
    return max(values) if values else None


def _finish(fig, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_privacy_utility(summary, out_path):
    """Accuracy and macro-F1 against the privacy budget, per mechanism.

    The chance band is drawn on both panels. Without it a reader cannot tell
    which of these curves belong to a model that learned anything at all, which
    is the central question for this cohort.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, metric, label in zip(axes, ('accuracy', 'f1_macro'), ('Accuracy', 'Macro F1')):
        for method, scope, name in MECHANISMS:
            epsilons, means, stds = sweep_series(summary, method, scope, metric)
            if not epsilons:
                continue
            ax.errorbar(epsilons, means, yerr=stds, marker='o', capsize=3,
                        color=COLOURS[f'{method}_{scope}'], label=name, linewidth=1.8)

        ceiling = best_non_private(summary, metric)
        if ceiling is not None:
            ax.axhline(ceiling, linestyle=':', color=COLOURS['reference'], linewidth=1.4,
                       label='Best non-private')

        if metric == 'accuracy':
            ax.axhline(summary.get('chance_accuracy', CHANCE_ACCURACY), linestyle='--',
                       color='black', linewidth=1.2, label='Chance')

        ax.set_xlabel('Privacy budget ε')
        ax.set_ylabel(label)
        ax.set_title(f'{label} against privacy budget')
        ax.grid(alpha=0.25)

    axes[0].legend(fontsize=8, loc='best')
    return _finish(fig, out_path)


def plot_scope_comparison(summary, out_path):
    """Full-model against head-only subject-level DP-FedAvg.

    This is the figure for the methodological finding: perturbing all ~25M
    parameters buries the update in noise, and restricting the mechanism to the
    6,147-parameter classifier head is what lets any signal survive.
    """
    epsilons = sorted({
        condition['epsilon'] for condition in summary['conditions']
        if condition['method'] == 'dpfedavg_userlevel' and condition['epsilon'] is not None
    })
    if not epsilons:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No subject-level DP runs recorded', ha='center', va='center')
        ax.axis('off')
        return _finish(fig, out_path)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.36
    positions = range(len(epsilons))

    for offset, scope, name in ((-width / 2, 'full', 'Full model (~25M params)'),
                                (width / 2, 'head', 'Head only (6,147 params)')):
        means, errors = [], []
        for epsilon in epsilons:
            condition = find_condition(summary, 'dpfedavg_userlevel', scope, epsilon)
            means.append((condition or {}).get('f1_macro_mean') or 0.0)
            errors.append((condition or {}).get('f1_macro_std') or 0.0)
        ax.bar([p + offset for p in positions], means, width, yerr=errors, capsize=3,
               label=name, color=COLOURS[f'dpfedavg_userlevel_{scope}'])

    ceiling = best_non_private(summary, 'f1_macro')
    if ceiling is not None:
        ax.axhline(ceiling, linestyle=':', color=COLOURS['reference'], linewidth=1.4,
                   label='Best non-private')

    ax.set_xticks(list(positions))
    ax.set_xticklabels([f'ε = {e:g}' for e in epsilons])
    ax.set_ylabel('Macro F1')
    ax.set_title('Subject-level DP-FedAvg: perturbing the whole model against the head only')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    return _finish(fig, out_path)


def plot_convergence(histories, out_path, title='Federated averaging convergence'):
    """Validation accuracy per round for each supplied history file."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    plotted = 0
    for label, path in histories:
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            history = json.load(handle)
        if not history.get('rounds'):
            continue
        ax.plot(history['rounds'], [a / 100 if a > 1 else a for a in history['accuracy']],
                marker='.', linewidth=1.6, label=label)
        plotted += 1

    ax.axhline(CHANCE_ACCURACY, linestyle='--', color='black', linewidth=1.2, label='Chance')
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Validation accuracy')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize=8)
    return _finish(fig, out_path)


def plot_cohort_comparison(rows, out_path):
    """Accuracy before and after the subject-level split fix.

    ``rows`` is ``[(label, accuracy), ...]``. The point of the figure is the size
    of the drop, so the chance line is drawn to show where the honest numbers sit.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    bars = ax.bar(labels, values, color=[COLOURS['reference'], COLOURS['centralised_dp_head']][:len(rows)]
                  if len(rows) <= 2 else None)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f'{value * 100:.1f}%',
                ha='center', fontsize=10)

    ax.axhline(CHANCE_ACCURACY, linestyle='--', color='black', linewidth=1.2, label='Chance')
    ax.set_ylabel('Test accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Effect of removing subject-level leakage')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    return _finish(fig, out_path)


def main():  # pragma: no cover - CLI wiring over the tested functions above
    parser = argparse.ArgumentParser(description='Render PPXFL publication figures')
    parser.add_argument('--results-dir', default='results_v2')
    parser.add_argument('--out-dir', default=os.path.join('..', 'figures'))
    args = parser.parse_args()

    metrics_dir = os.path.join(args.results_dir, 'metrics')
    summary = load_summary(os.path.join(metrics_dir, 'results_summary.json'))

    written = [
        plot_privacy_utility(summary, os.path.join(args.out_dir, 'privacy_utility_v2.png')),
        plot_scope_comparison(summary, os.path.join(args.out_dir, 'dp_scope_comparison_v2.png')),
        plot_convergence(
            [(f'seed {seed}',
              os.path.join(metrics_dir, f'resnet50_fedavg_K4_T20_E3_f0_s{seed}_v2_history.json'))
             for seed in (42, 123, 2024)],
            os.path.join(args.out_dir, 'fedavg_convergence_v2.png'),
        ),
        plot_convergence(
            [(f'ε = {eps:g}',
              os.path.join(
                  metrics_dir,
                  f'resnet50_dpfedavg_userlevel_head_T20_E3_f0_s42_eps{eps}_v2_history.json'))
             for eps in (2.0, 5.0, 10.0)],
            os.path.join(args.out_dir, 'dpfedavg_head_convergence_v2.png'),
            title='Head-scope subject-level DP-FedAvg convergence',
        ),
    ]
    for path in written:
        print(f'wrote {path}')


if __name__ == '__main__':  # pragma: no cover
    main()
