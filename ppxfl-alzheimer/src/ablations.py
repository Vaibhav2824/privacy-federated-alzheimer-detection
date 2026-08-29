"""
ablations.py — Ablation Study Runner
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Runs ablation experiments A1–A7 to isolate component contributions.
Each ablation changes exactly one variable.
"""

import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centralised_train import MRIDataset, compute_class_weights, compute_metrics, evaluate, train_one_epoch
from models import get_model
from partition import dirichlet_partition_subjects, expand_subjects_to_indices
from splits import load_split

# MIA is evaluated separately (see mia.py) with a proper shadow-model attack —
# running it per ablation cell would be expensive and add nothing an ablation
# needs (ablations isolate utility trade-offs; the dedicated DP/MIA sweep in
# Phase B covers privacy leakage far more rigorously).


def run_single_experiment(model_name, images, labels, manifest, train_idx, val_idx, test_idx,
                          device, epochs=20, lr=1e-4, batch_size=32, pretrained=True,
                          use_fl=False, num_clients=4, alpha=0.5, local_epochs=5,
                          seed=42):
    """
    Run a single training experiment with configurable settings.

    Returns:
        metrics: dict with accuracy, f1, auroc, etc.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    class_weights = compute_class_weights(labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if use_fl:
        return _run_fl_ablation(
            model_name, images, labels, manifest, train_idx, test_idx,
            device, epochs, lr, batch_size, num_clients, alpha,
            local_epochs, pretrained, class_weights, criterion, seed
        )

    # Centralised training
    model = get_model(model_name, num_classes=3, pretrained=pretrained).to(device)

    train_dataset = MRIDataset(images[train_idx], labels[train_idx], augment=True)
    MRIDataset(images[val_idx], labels[val_idx])
    test_dataset = MRIDataset(images[test_idx], labels[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _epoch in tqdm(range(epochs), desc='  Training', unit='ep', leave=False):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

    _, _, preds, true_labels, probs = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(true_labels, preds, probs)
    return metrics


def _run_fl_ablation(model_name, images, labels, manifest, train_idx, test_idx,
                     device, epochs, lr, batch_size, num_clients, alpha,
                     local_epochs, pretrained, class_weights, criterion, seed):
    """Simplified FL simulation for ablation without full Flower overhead.

    Partitions by SUBJECT (not slice) within the fold's train pool, same
    invariant as partition.py — a subject's scans must all land on one
    simulated client.
    """
    from collections import OrderedDict

    train_manifest = manifest[manifest['array_index'].isin(train_idx)]
    subj_table = train_manifest.drop_duplicates('subject_id')[['subject_id', 'label']]
    client_subjects = dirichlet_partition_subjects(
        subj_table['subject_id'].values, subj_table['label'].values,
        num_clients=num_clients, alpha=alpha, seed=seed,
    )
    client_indices = expand_subjects_to_indices(client_subjects, manifest)

    # Initialise global model
    global_model = get_model(model_name, num_classes=3, pretrained=pretrained).to(device)

    num_rounds = min(epochs, 20)

    for _round_num in range(num_rounds):
        collected_weights = []
        collected_sizes = []

        for cid in range(num_clients):
            local_model = get_model(model_name, num_classes=3, pretrained=False).to(device)
            local_model.load_state_dict(global_model.state_dict())

            c_idx = client_indices[cid]
            c_dataset = MRIDataset(images[c_idx], labels[c_idx], augment=True)
            c_loader = DataLoader(c_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)

            for _ in range(local_epochs):
                train_one_epoch(local_model, c_loader, criterion, optimizer, device)

            collected_weights.append({pn: pv.cpu().clone() for pn, pv in local_model.state_dict().items()})
            collected_sizes.append(len(c_idx))

        total_size = sum(collected_sizes)
        avg_state = OrderedDict()

        for key in global_model.state_dict().keys():
            avg_state[key] = sum(
                collected_weights[i][key].float() * (collected_sizes[i] / total_size)
                for i in range(num_clients)
            )

        global_model.load_state_dict(avg_state)

    test_dataset = MRIDataset(images[test_idx], labels[test_idx])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    _, _, preds, true_labels, probs = evaluate(global_model, test_loader, criterion, device)
    metrics = compute_metrics(true_labels, preds, probs)
    return metrics


def run_all_ablations(data_dir, splits_path, results_dir, fold=0, epochs=15, seed=42):
    """Run the 8-cell ablation suite (subject-level fold 0, held-out test never touched).

    K=4/E=3/alpha=0.5 is the main FedAvg experiment (B3) and is NOT re-run here;
    every cell below isolates exactly one change from that baseline.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images = np.load(os.path.join(data_dir, 'all_images.npy'))
    labels = np.load(os.path.join(data_dir, 'all_labels.npy'))
    manifest_path = os.path.join(data_dir, 'manifest.csv')
    manifest = pd.read_csv(manifest_path)
    split = load_split(fold, manifest_path, splits_path)
    train_idx, val_idx, test_idx = split['train_idx'], split['val_idx'], split['test_idx']
    print(f"  Fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} slices")

    results = {}

    fl_epochs = min(epochs, 8)
    ablation_configs = [
        ('A6_K2', 'K=2 clients', {'model_name': 'resnet50', 'epochs': fl_epochs, 'use_fl': True, 'num_clients': 2}),
        ('A6_K8', 'K=8 clients', {'model_name': 'resnet50', 'epochs': fl_epochs, 'use_fl': True, 'num_clients': 8}),
        ('A7_E1', 'E=1 local epoch', {'model_name': 'resnet50', 'epochs': fl_epochs, 'use_fl': True, 'local_epochs': 1}),
        ('A7_E5', 'E=5 local epochs', {'model_name': 'resnet50', 'epochs': fl_epochs, 'use_fl': True, 'local_epochs': 5}),
        ('A5_alpha0.1', 'Non-IID α=0.1', {'model_name': 'resnet50', 'epochs': fl_epochs, 'use_fl': True, 'alpha': 0.1}),
        ('A5_alpha100', 'IID (α=100)', {'model_name': 'resnet50', 'epochs': fl_epochs, 'use_fl': True, 'alpha': 100.0}),
        ('A8_vgg19_fl', 'VGG19 FedAvg (arch check)', {'model_name': 'vgg19', 'epochs': fl_epochs, 'use_fl': True}),
        ('A9_scratch', 'ResNet50 centralised, no ImageNet init', {'model_name': 'resnet50', 'epochs': epochs, 'pretrained': False}),
    ]

    overall_pbar = tqdm(ablation_configs, desc='Ablation Studies', unit='exp',
                        bar_format='{l_bar}{bar:30}{r_bar}')

    for key, desc, config in overall_pbar:
        overall_pbar.set_postfix_str(desc)
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"  {desc}")
        tqdm.write(f"{'='*60}")

        model_name = config.pop('model_name', 'resnet50')
        exp_epochs = config.pop('epochs', epochs)

        try:
            results[key] = run_single_experiment(
                model_name, images, labels, manifest, train_idx, val_idx, test_idx, device,
                epochs=exp_epochs, seed=seed, **config
            )
            tqdm.write(f"  ✓ {key}: Acc={results[key].get('accuracy',0)*100:.1f}%, "
                       f"F1={results[key].get('f1_macro',0):.3f}")
        except Exception as e:
            tqdm.write(f"  ✗ {key}: ERROR — {e}")
            import traceback
            traceback.print_exc()
            results[key] = {'error': str(e)}

        # Free GPU memory
        torch.cuda.empty_cache()

    # Save all results
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    # Convert to serialisable format
    serialisable = {}
    for k, v in results.items():
        serialisable[k] = {
            kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
            for kk, vv in v.items()
        }

    with open(os.path.join(metrics_dir, 'ablation_results.json'), 'w') as f:
        json.dump(serialisable, f, indent=2)

    # Create ablation table
    ablation_df = pd.DataFrame([
        {'Ablation': k,
         'Accuracy': f"{v.get('accuracy', 0)*100:.1f}%",
         'F1 (Macro)': f"{v.get('f1_macro', 0)*100:.1f}%",
         'AUROC': f"{v.get('auroc_macro', 0)*100:.1f}%"}
        for k, v in results.items()
    ])

    csv_path = os.path.join(metrics_dir, 'ablation_table.csv')
    ablation_df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Ablation table saved: {csv_path}")
    print(f"\n{ablation_df.to_string(index=False)}")

    # Generate ablation plots
    _plot_ablation_results(results, os.path.join(results_dir, 'figures'))

    return results


def _plot_ablation_results(results, save_dir):
    """Generate ablation visualisation plots."""
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Accuracy vs K (clients)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ablation Study Results', fontsize=14, fontweight='bold')

    # Clients ablation
    k_values = []
    k_accs = []
    for k, v in results.items():
        if k.startswith('A6_K'):
            K = int(k.split('K')[1])
            k_values.append(K)
            k_accs.append(v.get('accuracy', 0) * 100)

    if k_values:
        ax1.plot(k_values, k_accs, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clients (K)', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Accuracy vs Number of Clients')
        ax1.grid(True, alpha=0.3)

    # Local epochs ablation
    e_values = []
    e_accs = []
    for k, v in results.items():
        if k.startswith('A7_E'):
            E = int(k.split('E')[1])
            e_values.append(E)
            e_accs.append(v.get('accuracy', 0) * 100)

    if e_values:
        ax2.plot(e_values, e_accs, 'rs-', linewidth=2, markersize=8)
        ax2.set_xlabel('Local Epochs (E)', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Accuracy vs Local Epochs')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Alpha comparison
    alpha_values = []
    alpha_accs = []
    for k, v in results.items():
        if k.startswith('A5_alpha'):
            a = float(k.split('alpha')[1])
            alpha_values.append(a if a < 100 else 'IID')
            alpha_accs.append(v.get('accuracy', 0) * 100)

    if alpha_values:
        fig, ax = plt.subplots(figsize=(8, 5))
        x_labels = [str(a) for a in alpha_values]
        ax.bar(x_labels, alpha_accs, color=['#F44336', '#FF9800', '#4CAF50', '#2196F3'])
        ax.set_xlabel('Dirichlet α', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('IID vs Non-IID Performance', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ablation_alpha.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  ✓ Ablation plots saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='PPXFL Ablation Studies')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--splits-path', type=str, default=None)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'processed')
    if args.splits_path is None:
        args.splits_path = os.path.join(project_root, 'data', 'splits', 'splits_v1.json')
    results_dir = os.path.join(project_root, 'results')

    run_all_ablations(args.data_dir, args.splits_path, results_dir, args.fold, args.epochs, args.seed)


if __name__ == '__main__':
    main()
