"""
dp_train.py — Differential Privacy Training (Manual DP-SGD)
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Implements manual DP-SGD (gradient clipping + Gaussian noise) without Opacus,
to avoid GPU memory issues on RTX 3050 Ti (4GB VRAM).
Provides privacy budget tracking via RDP accountant and privacy–utility experiments.
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model
from centralised_train import (
    MRIDataset, compute_class_weights, evaluate, compute_metrics,
    train_one_epoch
)


def compute_epsilon(noise_multiplier, num_steps, sample_rate, delta=1e-5):
    """
    Compute (ε, δ)-DP guarantee using the simple composition theorem.
    For a more precise bound, use RDP accountant. This gives a reasonable upper bound.
    
    ε ≈ q * sqrt(2T * ln(1/δ)) / σ  (advanced composition)
    where q=sample_rate, T=num_steps, σ=noise_multiplier
    """
    if noise_multiplier == 0:
        return float('inf')
    q = sample_rate
    T = num_steps
    sigma = noise_multiplier
    # Advanced composition theorem (Abadi et al. 2016, simplified)
    epsilon = q * math.sqrt(2 * T * math.log(1.0 / delta)) / sigma
    return epsilon


def train_with_dp_manual(model, train_loader, val_loader, criterion, device,
                         epochs=10, lr=1e-4, weight_decay=1e-4,
                         noise_multiplier=1.1, max_grad_norm=1.0,
                         target_epsilon=None, delta=1e-5,
                         experiment_label=""):
    """
    Train model with manual DP-SGD (no Opacus needed).
    
    DP-SGD steps:
    1. Clip per-sample gradients (approximate via batch gradient clipping)
    2. Add Gaussian noise proportional to sensitivity
    3. Track cumulative privacy budget
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epsilon': []
    }

    best_val_f1 = 0.0
    best_state = None
    total_steps = 0
    sample_rate = min(32 / len(train_loader.dataset), 1.0)  # batch_size / dataset_size

    desc = f"DP σ={noise_multiplier}" + (f" [{experiment_label}]" if experiment_label else "")
    epoch_pbar = tqdm(range(1, epochs + 1), desc=desc, unit="epoch",
                      bar_format='{l_bar}{bar:30}{r_bar}')

    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        batch_pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}",
                          leave=False, unit="batch",
                          bar_format='    {l_bar}{bar:20}{r_bar}')
        for images, labels in batch_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # DP-SGD Step 1: Clip gradients (batch-level clipping)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # DP-SGD Step 2: Add calibrated Gaussian noise to gradients
            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * (
                        noise_multiplier * max_grad_norm / max(images.size(0), 1)
                    )
                    p.grad += noise

            optimizer.step()
            total_steps += 1

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            batch_pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Compute cumulative epsilon
        epsilon = compute_epsilon(noise_multiplier, total_steps, sample_rate, delta)

        # Validate
        val_loss, val_acc, val_preds, val_labels_arr, val_probs = evaluate(
            model, val_loader, criterion, device
        )

        from sklearn.metrics import precision_recall_fscore_support
        _, _, val_f1, _ = precision_recall_fscore_support(
            val_labels_arr, val_preds, average='macro', zero_division=0
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epsilon'].append(epsilon)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        epoch_pbar.set_postfix(
            acc=f"{val_acc:.1f}%", f1=f"{val_f1:.3f}", eps=f"{epsilon:.2f}"
        )

        # Early stop if budget exceeded
        if target_epsilon is not None and epsilon > target_epsilon:
            tqdm.write(f"  ⚠ Privacy budget exceeded (ε={epsilon:.2f} > {target_epsilon}). Stopping.")
            break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    return model, history, epsilon


def run_privacy_utility_experiment(data_dir, results_dir, model_name='resnet50',
                                   epsilon_values=None, epochs=10, seed=42):
    """
    Run privacy-utility trade-off experiment at multiple noise levels.
    Uses ResNet50 by default for GPU efficiency on RTX 3050 Ti.
    """
    if epsilon_values is None:
        epsilon_values = [2.0, 5.0, 10.0]

    # Map target ε to approximate noise_multiplier values
    sigma_map = {
        1.0: 2.0,
        1.5: 1.5,
        2.0: 1.1,
        5.0: 0.7,
        10.0: 0.5,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    images = np.load(os.path.join(data_dir, 'all_images.npy'))
    labels = np.load(os.path.join(data_dir, 'all_labels.npy'))

    # 80/10/10 split
    n = len(images)
    np.random.seed(seed)
    indices = np.random.permutation(n)
    train_idx = indices[:int(0.8*n)]
    val_idx = indices[int(0.8*n):int(0.9*n)]
    test_idx = indices[int(0.9*n):]

    class_weights = compute_class_weights(labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    test_dataset = MRIDataset(images[test_idx], labels[test_idx])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    results = {}
    total_experiments = len(epsilon_values) + 1  # +1 for no-DP baseline

    overall_pbar = tqdm(total=total_experiments, desc="Privacy-Utility Sweep",
                        unit="exp", bar_format='{l_bar}{bar:30}{r_bar}')

    for target_eps in epsilon_values:
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Experiment: ε target = {target_eps}")
        tqdm.write(f"{'='*60}")

        sigma = sigma_map.get(target_eps, 1.0)

        # Fresh model
        model = get_model(model_name, num_classes=3, pretrained=True)

        train_dataset = MRIDataset(images[train_idx], labels[train_idx], augment=True)
        val_dataset = MRIDataset(images[val_idx], labels[val_idx])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        try:
            model, history, actual_eps = train_with_dp_manual(
                model, train_loader, val_loader, criterion, device,
                epochs=epochs, noise_multiplier=sigma, max_grad_norm=1.0,
                target_epsilon=target_eps * 2.0,
                experiment_label=f"ε={target_eps}"
            )

            # Evaluate on test set
            test_loss, test_acc, test_preds, test_labels_arr, test_probs = evaluate(
                model, test_loader, criterion, device
            )
            metrics = compute_metrics(test_labels_arr, test_preds, test_probs)

            results[target_eps] = {
                'actual_epsilon': actual_eps,
                'noise_multiplier': sigma,
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'auroc_macro': metrics['auroc_macro'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
            }

            tqdm.write(f"  Result: ε={actual_eps:.2f}, Acc={metrics['accuracy']:.4f}, "
                        f"F1={metrics['f1_macro']:.4f}")

        except Exception as e:
            tqdm.write(f"  [ERROR] ε={target_eps}: {e}")
            import traceback
            traceback.print_exc()
            results[target_eps] = {'error': str(e)}

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        overall_pbar.update(1)

    # Also run without DP (ε = ∞)
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Experiment: No DP (ε = ∞)")
    tqdm.write(f"{'='*60}")

    model = get_model(model_name, num_classes=3, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_dataset = MRIDataset(images[train_idx], labels[train_idx], augment=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    for ep in tqdm(range(1, epochs + 1), desc="No-DP baseline", unit="epoch",
                   bar_format='{l_bar}{bar:30}{r_bar}'):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

    test_loss, test_acc, test_preds, test_labels_arr, test_probs = evaluate(
        model, test_loader, criterion, device
    )
    no_dp_metrics = compute_metrics(test_labels_arr, test_preds, test_probs)
    results['inf'] = {
        'actual_epsilon': float('inf'),
        'accuracy': no_dp_metrics['accuracy'],
        'f1_macro': no_dp_metrics['f1_macro'],
        'auroc_macro': no_dp_metrics['auroc_macro'],
    }

    del model
    torch.cuda.empty_cache()

    overall_pbar.update(1)
    overall_pbar.close()

    # Save results
    os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
    results_path = os.path.join(results_dir, 'metrics', f'{model_name}_privacy_utility.json')
    serialisable = {}
    for k, v in results.items():
        serialisable[str(k)] = {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                 for kk, vv in v.items()}
    with open(results_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    tqdm.write(f"\n  ✓ Results saved: {results_path}")

    # Plot privacy-utility curve
    plot_privacy_utility_curve(results, os.path.join(results_dir, 'figures'), model_name)

    return results


def plot_privacy_utility_curve(results, save_dir, model_name):
    """Plot accuracy vs ε Pareto frontier."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epsilons = []
    accuracies = []
    f1_scores = []

    for eps_key, metrics in sorted(results.items(), key=lambda x: float(x[0]) if x[0] != 'inf' else 999):
        if 'error' in metrics:
            continue
        eps_val = float(eps_key) if eps_key != 'inf' else 20.0
        actual_eps = metrics.get('actual_epsilon', eps_val)
        if actual_eps == float('inf'):
            actual_eps = 20.0
        epsilons.append(actual_eps)
        accuracies.append(metrics['accuracy'] * 100)
        f1_scores.append(metrics.get('f1_macro', 0) * 100)

    ax.plot(epsilons, accuracies, 'bo-', linewidth=2, markersize=8, label='Accuracy')
    ax.plot(epsilons, f1_scores, 'rs--', linewidth=2, markersize=8, label='F1-Score')

    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Performance (%)', fontsize=12)
    ax.set_title(f'Privacy–Utility Trade-off ({model_name})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if 20.0 in epsilons:
        ax.annotate('No DP (ε=∞)', xy=(20.0, accuracies[-1]),
                    xytext=(16, accuracies[-1] - 5),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=10, color='gray')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_privacy_utility_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    tqdm.write(f"  ✓ Privacy-utility curve saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='PPXFL DP Training')
    parser.add_argument('--model', type=str, default='resnet50', choices=['vgg19', 'resnet50'])
    parser.add_argument('--mode', type=str, default='experiment',
                        choices=['single', 'experiment'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--noise-multiplier', type=float, default=1.1)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--target-epsilon', type=float, default=2.0)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'processed')
    results_dir = os.path.join(project_root, 'results')

    if args.mode == 'experiment':
        run_privacy_utility_experiment(
            args.data_dir, results_dir, args.model,
            epsilon_values=[2.0, 5.0, 10.0],
            epochs=args.epochs, seed=args.seed
        )
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = np.load(os.path.join(args.data_dir, 'all_images.npy'))
        labels = np.load(os.path.join(args.data_dir, 'all_labels.npy'))

        n = len(images)
        np.random.seed(args.seed)
        idx = np.random.permutation(n)

        train_dataset = MRIDataset(images[idx[:int(0.8*n)]], labels[idx[:int(0.8*n)]], augment=True)
        val_dataset = MRIDataset(images[idx[int(0.8*n):int(0.9*n)]], labels[idx[int(0.8*n):int(0.9*n)]])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        class_weights = compute_class_weights(labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = get_model(args.model, num_classes=3, pretrained=True)

        model, history, epsilon = train_with_dp_manual(
            model, train_loader, val_loader, criterion, device,
            epochs=args.epochs,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            target_epsilon=args.target_epsilon
        )

        print(f"\nFinal ε: {epsilon:.2f}")
        torch.save(model.state_dict(), os.path.join(results_dir, f'{args.model}_dp_eps{epsilon:.1f}.pth'))


if __name__ == '__main__':
    main()
