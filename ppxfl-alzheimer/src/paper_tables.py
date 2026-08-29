"""paper_tables.py — Generate the paper's results tables from the results JSONs.

Every numeric table in the paper is written by this module into marker-delimited
blocks in ``paper.tex``:

    % BEGIN AUTO:centralised
    ... generated rows ...
    % END AUTO:centralised

Nothing between those markers is edited by hand, which is what makes the
number-coherence check in ``check_paper_numbers.py`` meaningful: a figure in the
paper and the corresponding entry in ``results_summary.json`` cannot drift apart
without this module being re-run.

Usage:
    python -m src.paper_tables --results-dir results_v2 --paper ../paper.tex
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import CHANCE_ACCURACY  # noqa: E402

MODEL_NAMES = {'resnet50': 'ResNet50', 'vgg19': 'VGG19'}


def fmt(mean, std, scale=1.0, digits=1):
    """``mean $\\pm$ std`` at a fixed precision, or an en-rule when absent."""
    if mean is None:
        return '---'
    body = f'{mean * scale:.{digits}f}'
    if std is None:
        return body
    return f'{body} $\\pm$ {std * scale:.{digits}f}'


def select(conditions, method, scope=None, epsilon=None, model='resnet50'):
    """The single condition matching a full experimental specification."""
    for condition in conditions:
        if (condition['method'] == method
                and condition['dp_scope'] == scope
                and condition['epsilon'] == epsilon
                and condition['model'] == model):
            return condition
    return None


def metric_cells(condition):
    """Accuracy, macro F1 and macro AUROC cells for one condition."""
    if condition is None:
        return ['---', '---', '---', '0']
    return [
        fmt(condition['accuracy_mean'], condition['accuracy_std'], 100, 1),
        fmt(condition['f1_macro_mean'], condition['f1_macro_std'], 1, 3),
        fmt(condition['auroc_macro_mean'], condition['auroc_macro_std'], 1, 3),
        str(condition['n_runs']),
    ]


def centralised_rows(summary):
    """One row per architecture, non-private centralised training."""
    rows = []
    for model in ('vgg19', 'resnet50'):
        condition = select(summary['conditions'], 'centralised', model=model)
        if condition is None:
            continue
        rows.append(' & '.join([MODEL_NAMES[model], *metric_cells(condition)]) + r' \\')
    return '\n'.join(rows)


def federated_rows(summary):
    """Centralised against federated averaging, same architecture."""
    rows = []
    centralised = select(summary['conditions'], 'centralised')
    federated = select(summary['conditions'], 'fedavg')
    if centralised is not None:
        rows.append(' & '.join(['Centralised ResNet50', *metric_cells(centralised)]) + r' \\')
    if federated is not None:
        rows.append(' & '.join([r'FedAvg ResNet50 ($K{=}4$)', *metric_cells(federated)]) + r' \\')
    return '\n'.join(rows)


def dp_sweep_rows(summary):
    """Sample-level DP-SGD across the budget sweep, with the non-private row."""
    rows = []
    baseline = select(summary['conditions'], 'centralised')
    if baseline is not None:
        rows.append(' & '.join([r'No DP ($\varepsilon = \infty$)', *metric_cells(baseline)]) + r' \\')
        rows.append(r'\midrule')
    for epsilon in (2.0, 5.0, 10.0):
        condition = select(summary['conditions'], 'centralised_dp', 'head', epsilon)
        if condition is None:
            continue
        rows.append(' & '.join([f'$\\varepsilon = {epsilon:g}$', *metric_cells(condition)]) + r' \\')
    return '\n'.join(rows)


def userlevel_rows(summary):
    """Full-model against head-only subject-level DP-FedAvg, per budget.

    The perturbed-parameter count is the column that explains the rest of the
    row: the Gaussian mechanism's noise norm grows with the dimension it is
    applied to, so a 4{,}000-fold reduction in that dimension is the mechanism's
    entire difference.
    """
    rows = []
    baseline = select(summary['conditions'], 'fedavg')
    if baseline is not None:
        cells = metric_cells(baseline)
        rows.append(' & '.join([r'FedAvg, no DP', '---', *cells]) + r' \\')
        rows.append(r'\midrule')

    data_rows = 0
    for scope, label in (('full', 'Full model'), ('head', 'Head only')):
        for epsilon in (2.0, 5.0, 10.0):
            condition = select(summary['conditions'], 'dpfedavg_userlevel', scope, epsilon)
            if condition is None:
                continue
            perturbed = condition.get('perturbed_params')
            perturbed_cell = f'{perturbed:,}'.replace(',', '{,}') if perturbed else '---'
            rows.append(
                ' & '.join([f'{label}, $\\varepsilon = {epsilon:g}$', perturbed_cell,
                            *metric_cells(condition)]) + r' \\')
            data_rows += 1
        rows.append(r'\midrule')

    # A trailing separator with nothing after it renders as a stray rule.
    while rows and rows[-1] == r'\midrule':
        rows.pop()
    return '\n'.join(rows) if data_rows else ''


def significance_rows(stats):
    """Paired Wilcoxon rows for each private condition against its baseline."""
    rows = []
    for comparison in stats.get('paired_wilcoxon_dp_vs_nondp', []):
        if 'p_value' not in comparison or 'F1' not in comparison.get('comparison', ''):
            continue
        label = comparison['comparison'].split(' vs ')[-1].replace('resnet50 | ', '')
        label = label.replace(' F1', '').replace('_', r'\_').replace('|', 'x')
        rows.append(
            f"{label} & {comparison['n_pairs']} & "
            f"{comparison.get('mean_a', float('nan')):.3f} & "
            f"{comparison.get('mean_b', float('nan')):.3f} & "
            f"{comparison['p_value']:.3f} & "
            f"{'yes' if comparison.get('significant_at_0.05') else 'no'}" + r' \\')
    return '\n'.join(rows)


def cohort_facts(summary):
    """Single-value macros the prose reads from, so prose cannot drift either."""
    total_runs = sum(condition['n_runs'] for condition in summary['conditions'])
    best = max(
        (c for c in summary['conditions']
         if c['epsilon'] is None and c['accuracy_mean'] is not None),
        key=lambda c: c['accuracy_mean'], default=None)
    lines = [
        rf'\newcommand{{\ChanceAccuracy}}{{{summary.get("chance_accuracy", CHANCE_ACCURACY) * 100:.1f}\%}}',
        rf'\newcommand{{\TotalRuns}}{{{total_runs}}}',
        rf'\newcommand{{\NumConditions}}{{{summary["n_conditions"]}}}',
    ]
    if best is not None:
        lines.append(rf'\newcommand{{\BestNonPrivateAcc}}{{{best["accuracy_mean"] * 100:.1f}\%}}')
        lines.append(rf'\newcommand{{\BestNonPrivateF}}{{{best["f1_macro_mean"]:.3f}}}')
    return '\n'.join(lines)


BLOCKS = {
    'centralised': centralised_rows,
    'federated': federated_rows,
    'dp_sweep': dp_sweep_rows,
    'userlevel': userlevel_rows,
    'facts': cohort_facts,
}


def splice(text, name, body):
    """Replace the content between the markers for ``name``.

    Raises if the markers are absent: silently skipping would let the paper keep
    stale numbers while this module reported success.
    """
    begin = f'% BEGIN AUTO:{name}'
    end = f'% END AUTO:{name}'
    if begin not in text or end not in text:
        raise ValueError(f'paper.tex has no "{begin}" / "{end}" block')
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return f'{head}{begin}\n{body}\n{end}{tail}'


def render(paper_text, summary, stats):
    """Apply every generated block to the paper source."""
    for name, builder in BLOCKS.items():
        paper_text = splice(paper_text, name, builder(summary))
    if '% BEGIN AUTO:significance' in paper_text:
        paper_text = splice(paper_text, 'significance', significance_rows(stats))
    return paper_text


def main():  # pragma: no cover - CLI wiring over the tested functions above
    parser = argparse.ArgumentParser(description='Write the paper tables from the results JSONs')
    parser.add_argument('--results-dir', default='results_v2')
    parser.add_argument('--paper', default=os.path.join('..', 'paper.tex'))
    args = parser.parse_args()

    metrics_dir = os.path.join(args.results_dir, 'metrics')
    with open(os.path.join(metrics_dir, 'results_summary.json')) as handle:
        summary = json.load(handle)
    stats_path = os.path.join(metrics_dir, 'stats_analysis.json')
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

    text = open(args.paper, encoding='utf-8').read()
    open(args.paper, 'w', encoding='utf-8', newline='\n').write(render(text, summary, stats))
    print(f'regenerated {len(BLOCKS)} table blocks in {args.paper}')


if __name__ == '__main__':  # pragma: no cover
    main()
