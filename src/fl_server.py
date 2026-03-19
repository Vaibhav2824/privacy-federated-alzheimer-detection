"""
fl_server.py — Federated Learning Server (Manual FedAvg Simulation)
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Runs FL simulation using a manual FedAvg loop (no ray dependency).
Supports configurable clients, rounds, local epochs, and optional DP-SGD.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model
from centralised_train import (
    MRIDataset, compute_class_weights, evaluate, compute_metrics,
    train_one_epoch
)


def fedavg_aggregate(client_weights, client_sizes):
    """
    FedAvg: weighted average of client model parameters.

    Args:
        client_weights: list of state_dicts (on CPU)
        client_sizes: list of int (num samples per client)

    Returns:
        averaged state_dict
    """
    total = sum(client_sizes)
    avg_state = OrderedDict()

    for key in client_weights[0].keys():
        avg_state[key] = sum(
            client_weights[i][key].float() * (client_sizes[i] / total)
            for i in range(len(client_weights))
        )

    return avg_state


def run_simulation(model_name='vgg19', num_clients=4, num_rounds=50,
                   local_epochs=5, batch_size=32, lr=1e-4,
                   dp_enabled=False, noise_multiplier=1.1, max_grad_norm=1.0,
                   data_dir=None, results_dir=None, seed=42):
    """
    Run federated learning simulation using manual FedAvg.

    Args:
        model_name: 'vgg19' or 'resnet50'
        num_clients: K — number of clients
        num_rounds: T — total FL rounds
        local_epochs: E — local epochs per round
        batch_size: Training batch size
        lr: Learning rate
        dp_enabled: Enable DP-SGD (per-client)
        noise_multiplier: σ for DP
        max_grad_norm: C for gradient clipping
        data_dir: Path to clients/ directory
        results_dir: Path to save results
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(project_root, 'data', 'clients')
    if results_dir is None:
        results_dir = os.path.join(project_root, 'results')

    figures_dir = os.path.join(results_dir, 'figures')
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # ── Load client datasets ────────────────────────────────────────────
    client_loaders = []
    client_sizes = []
    for cid in range(1, num_clients + 1):
        cdir = os.path.join(data_dir, f'client_{cid}')
        imgs = np.load(os.path.join(cdir, 'images.npy'))
        lbls = np.load(os.path.join(cdir, 'labels.npy'))
        ds = MRIDataset(imgs, lbls, augment=True)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)
        client_sizes.append(len(ds))
        print(f"  Client {cid}: {len(ds)} samples  "
              f"(CN={np.sum(lbls==0)}, MCI={np.sum(lbls==1)}, AD={np.sum(lbls==2)})")

    # ── Server-side test set (10% held-out from full data) ──────────────
    processed_dir = os.path.join(os.path.dirname(data_dir), 'processed')
    all_images = np.load(os.path.join(processed_dir, 'all_images.npy'))
    all_labels = np.load(os.path.join(processed_dir, 'all_labels.npy'))

    np.random.seed(seed)
    n = len(all_images)
    perm = np.random.permutation(n)
    test_idx = perm[int(0.9 * n):]
    test_dataset = MRIDataset(all_images[test_idx], all_labels[test_idx])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(all_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Initialise global model ─────────────────────────────────────────
    global_model = get_model(model_name, num_classes=3, pretrained=True)

    dp_suffix = f"_dp_nm{noise_multiplier}" if dp_enabled else ""
    experiment_name = (f"{model_name}_fedavg_K{num_clients}_T{num_rounds}"
                       f"_E{local_epochs}{dp_suffix}")

    print(f"\n{'='*60}")
    print(f"Starting FL Simulation: {experiment_name}")
    print(f"  Model: {model_name} | Clients: {num_clients} | Rounds: {num_rounds}")
    print(f"  Local Epochs: {local_epochs} | Batch: {batch_size} | LR: {lr}")
    if dp_enabled:
        print(f"  DP: noise_mult={noise_multiplier}, max_grad_norm={max_grad_norm}")
    print(f"{'='*60}\n")

    # ── FL round loop ───────────────────────────────────────────────────
    round_metrics = {
        'rounds': [], 'accuracy': [], 'f1_macro': [],
        'auroc_macro': [], 'loss': [],
    }

    best_f1 = 0.0
    best_state = None

    for rnd in range(1, num_rounds + 1):
        collected_weights = []
        collected_sizes = []

        for cid in range(num_clients):
            # Clone global model → local model
            local_model = deepcopy(global_model).to(device)
            local_optimizer = optim.Adam(local_model.parameters(),
                                         lr=lr, weight_decay=1e-4)

            # Optional DP wrapping
            if dp_enabled:
                try:
                    from opacus import PrivacyEngine
                    from opacus.validators import ModuleValidator
                    if not ModuleValidator.is_valid(local_model):
                        local_model = ModuleValidator.fix(local_model).to(device)
                    local_optimizer = optim.Adam(local_model.parameters(),
                                                 lr=lr, weight_decay=1e-4)
                    pe = PrivacyEngine()
                    local_model, local_optimizer, loader = pe.make_private(
                        module=local_model,
                        optimizer=local_optimizer,
                        data_loader=client_loaders[cid],
                        noise_multiplier=noise_multiplier,
                        max_grad_norm=max_grad_norm,
                    )
                except Exception as e:
                    print(f"  [DP-warn] Client {cid+1}: {e}")
                    loader = client_loaders[cid]
            else:
                loader = client_loaders[cid]

            # Local training for E epochs
            for _ in range(local_epochs):
                train_one_epoch(local_model, loader, criterion,
                                local_optimizer, device)

            # Collect weights (move to CPU)
            state = OrderedDict()
            for pname, pval in local_model.state_dict().items():
                # Opacus may prepend '_module.' to parameter names
                clean_name = pname.replace('_module.', '')
                state[clean_name] = pval.cpu().clone()
            collected_weights.append(state)
            collected_sizes.append(client_sizes[cid])

        # FedAvg aggregation
        global_state = fedavg_aggregate(collected_weights, collected_sizes)
        global_model.load_state_dict(global_state)

        # ── Server-side evaluation ──────────────────────────────────────
        global_model.to(device)
        loss, acc, preds, labels, probs = evaluate(
            global_model, test_loader, criterion, device
        )
        metrics = compute_metrics(labels, preds, probs)
        global_model.cpu()   # free GPU for next round

        round_metrics['rounds'].append(rnd)
        round_metrics['loss'].append(float(loss))
        round_metrics['accuracy'].append(float(acc))
        round_metrics['f1_macro'].append(float(metrics['f1_macro']))
        round_metrics['auroc_macro'].append(float(metrics['auroc_macro']))

        f1 = metrics['f1_macro']
        if f1 > best_f1:
            best_f1 = f1
            best_state = deepcopy(global_model.state_dict())

        star = " ★" if f1 >= best_f1 else ""
        print(f"  Round {rnd:3d}/{num_rounds} | Loss: {loss:.4f} "
              f"Acc: {acc:.1f}% | F1: {f1:.3f} AUROC: {metrics['auroc_macro']:.3f}{star}")

    # ── Save best global model ──────────────────────────────────────────
    if best_state is not None:
        global_model.load_state_dict(best_state)
    model_path = os.path.join(results_dir, f'{experiment_name}_global.pth')
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'experiment': experiment_name,
        'best_f1': best_f1,
    }, model_path)
    print(f"\n  ✓ Global model saved: {model_path}")

    # ── Final test evaluation ───────────────────────────────────────────
    global_model.to(device)
    loss, acc, preds, labels, probs = evaluate(
        global_model, test_loader, criterion, device
    )
    final_metrics = compute_metrics(labels, preds, probs)

    # Save metrics
    fl_metrics = {
        'experiment': experiment_name,
        'accuracy': float(final_metrics['accuracy']),
        'f1_macro': float(final_metrics['f1_macro']),
        'auroc_macro': float(final_metrics['auroc_macro']),
        'precision_macro': float(final_metrics['precision_macro']),
        'recall_macro': float(final_metrics['recall_macro']),
        'num_rounds': num_rounds,
        'num_clients': num_clients,
        'local_epochs': local_epochs,
        'dp_enabled': dp_enabled,
    }
    metrics_path = os.path.join(metrics_dir, f'{experiment_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(fl_metrics, f, indent=2)
    print(f"  ✓ Metrics saved: {metrics_path}")

    # Save round history
    history_path = os.path.join(metrics_dir, f'{experiment_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(round_metrics, f, indent=2)
    print(f"  ✓ Round history saved: {history_path}")

    # ── Convergence plot ────────────────────────────────────────────────
    plot_fl_convergence(round_metrics, figures_dir, experiment_name)

    print(f"\n{'='*60}")
    print(f"FL Simulation complete: {experiment_name}")
    print(f"  Final — Acc: {acc:.1f}%  F1: {final_metrics['f1_macro']:.3f}  "
          f"AUROC: {final_metrics['auroc_macro']:.3f}")
    print(f"{'='*60}")

    return round_metrics, fl_metrics


def plot_fl_convergence(round_metrics, save_dir, experiment_name):
    """Plot FL accuracy and loss vs round."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'FL Convergence — {experiment_name}',
                 fontsize=14, fontweight='bold')

    rounds = round_metrics['rounds']

    # Accuracy + F1
    ax1.plot(rounds, round_metrics['accuracy'], 'b-o', markersize=3,
             linewidth=2, label='Accuracy (%)')
    ax1.plot(rounds, [f * 100 for f in round_metrics['f1_macro']],
             'g--s', markersize=3, linewidth=1.5, label='F1 × 100')
    ax1.set_xlabel('FL Round', fontsize=12)
    ax1.set_ylabel('Performance', fontsize=12)
    ax1.set_title('Accuracy & F1 vs Round')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(rounds, round_metrics['loss'], 'r-o', markersize=3, linewidth=2)
    ax2.set_xlabel('FL Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Server Loss vs Round')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{experiment_name}_convergence.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Convergence plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='PPXFL FL Server / Simulation')
    parser.add_argument('--model', type=str, default='vgg19',
                        choices=['vgg19', 'resnet50'])
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--clients', type=int, default=4)
    parser.add_argument('--local-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dp-enabled', action='store_true')
    parser.add_argument('--noise-multiplier', type=float, default=1.1)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_simulation(
        model_name=args.model,
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dp_enabled=args.dp_enabled,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
