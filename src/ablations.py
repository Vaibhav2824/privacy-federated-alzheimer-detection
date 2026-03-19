"""
ablations.py — Ablation Study Runner
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Runs ablation experiments A1–A7 to isolate component contributions.
Each ablation changes exactly one variable.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model
from centralised_train import (
    MRIDataset, compute_class_weights, evaluate, compute_metrics,
    train_one_epoch
)
from partition import dirichlet_partition
from evaluate import membership_inference_attack


def run_single_experiment(model_name, images, labels, train_idx, val_idx, test_idx,
                          device, epochs=20, lr=1e-4, batch_size=32,
                          dp_enabled=False, noise_multiplier=1.1, max_grad_norm=1.0,
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
        # Simple FL simulation (without Flower for speed)
        return _run_fl_ablation(
            model_name, images, labels, train_idx, test_idx,
            device, epochs, lr, batch_size, num_clients, alpha,
            local_epochs, dp_enabled, noise_multiplier, max_grad_norm,
            class_weights, criterion, seed
        )
    
    # Centralised training
    model = get_model(model_name, num_classes=3, pretrained=True).to(device)
    
    train_dataset = MRIDataset(images[train_idx], labels[train_idx], augment=True)
    val_dataset = MRIDataset(images[val_idx], labels[val_idx])
    test_dataset = MRIDataset(images[test_idx], labels[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    if dp_enabled:
        from dp_train import train_with_dp_manual
        model, history, epsilon = train_with_dp_manual(
            model, train_loader, val_loader, criterion, device,
            epochs=epochs, lr=lr, noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in tqdm(range(epochs), desc='  Training', unit='ep', leave=False):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # Evaluate
    _, _, preds, true_labels, probs = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(true_labels, preds, probs)
    
    # Run MIA
    mia_acc, mia_adv, _ = membership_inference_attack(
        model, images[train_idx], labels[train_idx],
        images[test_idx], labels[test_idx], device
    )
    metrics['mia_accuracy'] = mia_acc
    metrics['mia_advantage'] = mia_adv
    
    return metrics


def _run_fl_ablation(model_name, images, labels, train_idx, test_idx,
                     device, epochs, lr, batch_size, num_clients, alpha,
                     local_epochs, dp_enabled, noise_multiplier, max_grad_norm,
                     class_weights, criterion, seed):
    """Simplified FL simulation for ablation without full Flower overhead."""
    from collections import OrderedDict
    
    # Partition training data
    train_labels = labels[train_idx]
    client_indices = dirichlet_partition(train_labels, num_clients, alpha, seed)
    
    # Initialise global model
    global_model = get_model(model_name, num_classes=3, pretrained=True).to(device)
    
    num_rounds = min(epochs, 20)
    
    for round_num in range(num_rounds):
        collected_weights = []
        collected_sizes = []
        
        for cid in range(num_clients):
            # Clone global model
            local_model = get_model(model_name, num_classes=3, pretrained=False).to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            # Get client data
            c_idx = train_idx[client_indices[cid]]
            c_dataset = MRIDataset(images[c_idx], labels[c_idx], augment=True)
            c_loader = DataLoader(c_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)
            
            # Local training
            for _ in range(local_epochs):
                train_one_epoch(local_model, c_loader, criterion, optimizer, device)
            
            collected_weights.append({pn: pv.cpu().clone() for pn, pv in local_model.state_dict().items()})
            collected_sizes.append(len(c_idx))
        
        # FedAvg aggregation
        total_size = sum(collected_sizes)
        avg_state = OrderedDict()
        
        for key in global_model.state_dict().keys():
            avg_state[key] = sum(
                collected_weights[i][key].float() * (collected_sizes[i] / total_size)
                for i in range(num_clients)
            )
        
        global_model.load_state_dict(avg_state)
    
    # Evaluate global model
    test_dataset = MRIDataset(images[test_idx], labels[test_idx])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    _, _, preds, true_labels, probs = evaluate(global_model, test_loader, criterion, device)
    metrics = compute_metrics(true_labels, preds, probs)
    
    # MIA
    mia_acc, mia_adv, _ = membership_inference_attack(
        global_model, images[train_idx], labels[train_idx],
        images[test_idx], labels[test_idx], device
    )
    metrics['mia_accuracy'] = mia_acc
    metrics['mia_advantage'] = mia_adv
    
    return metrics


def run_all_ablations(data_dir, results_dir, epochs=15, seed=42):
    """Run all ablation studies A1–A8."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    images = np.load(os.path.join(data_dir, 'all_images.npy'))
    labels = np.load(os.path.join(data_dir, 'all_labels.npy'))
    
    n = len(images)
    np.random.seed(seed)
    idx = np.random.permutation(n)
    train_idx = idx[:int(0.8*n)]
    val_idx = idx[int(0.8*n):int(0.9*n)]
    test_idx = idx[int(0.9*n):]
    
    results = {}
    
    # Define all ablation experiments
    ablation_configs = [
        ('A1_no_dp', 'A1: No DP (FL only)',
         dict(model_name='resnet50', epochs=epochs, use_fl=True, dp_enabled=False)),
        ('A2_with_dp', 'A2: Centralised + DP (ε≈2.0)',
         dict(model_name='resnet50', epochs=epochs, dp_enabled=True, noise_multiplier=1.1)),
        ('A4_resnet50', 'A4: ResNet50 centralised',
         dict(model_name='resnet50', epochs=epochs)),
        ('A4_vgg19', 'A4: VGG19 centralised',
         dict(model_name='vgg19', epochs=epochs)),
        ('A5_alpha0.1', 'A5: Non-IID α=0.1',
         dict(model_name='resnet50', epochs=min(epochs, 8), use_fl=True, alpha=0.1)),
        ('A5_alpha0.5', 'A5: Non-IID α=0.5',
         dict(model_name='resnet50', epochs=min(epochs, 8), use_fl=True, alpha=0.5)),
        ('A5_alpha1.0', 'A5: Non-IID α=1.0',
         dict(model_name='resnet50', epochs=min(epochs, 8), use_fl=True, alpha=1.0)),
        ('A5_alpha100.0', 'A5: IID (α=100)',
         dict(model_name='resnet50', epochs=min(epochs, 8), use_fl=True, alpha=100.0)),
        ('A6_K2', 'A6: K=2 clients',
         dict(model_name='resnet50', epochs=min(epochs, 8), use_fl=True, num_clients=2)),
        ('A6_K4', 'A6: K=4 clients',
         dict(model_name='resnet50', epochs=min(epochs, 8), use_fl=True, num_clients=4)),
        ('A6_K6', 'A6: K=6 clients',
         dict(model_name='resnet50', epochs=min(epochs, 8), use_fl=True, num_clients=6)),
        ('A7_E1', 'A7: E=1 local epochs',
         dict(model_name='resnet50', epochs=min(epochs, 8), use_fl=True, local_epochs=1)),
        ('A7_E3', 'A7: E=3 local epochs',
         dict(model_name='resnet50', epochs=min(epochs, 8), use_fl=True, local_epochs=3)),
        ('A7_E5', 'A7: E=5 local epochs',
         dict(model_name='resnet50', epochs=min(epochs, 8), use_fl=True, local_epochs=5)),
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
                model_name, images, labels, train_idx, val_idx, test_idx, device,
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
         'AUROC': f"{v.get('auroc_macro', 0)*100:.1f}%",
         'MIA Acc': f"{v.get('mia_accuracy', 0):.1f}%",
         'MIA Adv': f"{v.get('mia_advantage', 0):.1f}%"}
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
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'processed')
    results_dir = os.path.join(project_root, 'results')
    
    run_all_ablations(args.data_dir, results_dir, args.epochs, args.seed)


if __name__ == '__main__':
    main()
