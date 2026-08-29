"""
evaluate.py — Results Compilation
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Compiles per-run metrics JSONs into a single comparison table/plot.
For Membership Inference Attacks, see mia.py.
"""

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def run_all_experiments(data_dir, results_dir, seed=42):
    """
    Compile every per-run metrics JSON in results_dir into one comparison table.

    This function reads recorded results only; it does not train or evaluate a
    model, so it needs neither a device nor torch. An earlier version opened
    with a torch.device(...) call that torch was never imported for, which would
    have raised NameError on the first line had it been called.
    """
    metrics_dir = os.path.join(results_dir, 'metrics')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    all_results = []

    # Check what models/results already exist
    existing_metrics = {}
    for f in os.listdir(metrics_dir):
        if f.endswith('_metrics.json'):
            with open(os.path.join(metrics_dir, f)) as fp:
                existing_metrics[f] = json.load(fp)

    # Compile results from all experiments
    for fname, metrics in existing_metrics.items():
        result = {
            'experiment': fname.replace('_metrics.json', ''),
            'accuracy': metrics.get('accuracy', 0),
            'precision_macro': metrics.get('precision_macro', 0),
            'recall_macro': metrics.get('recall_macro', 0),
            'f1_macro': metrics.get('f1_macro', 0),
            'auroc_macro': metrics.get('auroc_macro', 0),
        }
        all_results.append(result)

    # Save comprehensive results table
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(metrics_dir, 'all_experiments_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n  ✓ Results table saved: {csv_path}")
        print(f"\n{df.to_string(index=False)}")

    return all_results


def generate_comparison_plots(results_dir):
    """Generate comparison plots from all experiment results."""
    metrics_dir = os.path.join(results_dir, 'metrics')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = os.path.join(metrics_dir, 'all_experiments_results.csv')
    if not os.path.exists(csv_path):
        print("  No results CSV found. Run experiments first.")
        return

    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Experiment Comparison', fontsize=14, fontweight='bold')

    metrics_to_plot = ['accuracy', 'f1_macro', 'auroc_macro']
    titles = ['Accuracy', 'F1-Score (Macro)', 'AUROC (Macro)']
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        if metric in df.columns:
            ax.bar(range(len(df)), df[metric] * 100, color=colors)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['experiment'], rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(f'{title} (%)')
            ax.set_title(title)
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'experiment_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comparison plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='PPXFL Evaluation (results compilation only — see mia.py for MIA)')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'compare'],
                        help='all: compile results | compare: generate plots')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'processed')
    results_dir = os.path.join(project_root, 'results')

    if args.experiment == 'all':
        run_all_experiments(args.data_dir, results_dir, args.seed)
        generate_comparison_plots(results_dir)
    elif args.experiment == 'compare':
        generate_comparison_plots(results_dir)


if __name__ == '__main__':
    main()
