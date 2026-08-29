"""
dp_train.py — Differential Privacy Training (real per-sample DP-SGD via Opacus)
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

The previous version of this file clipped the BATCH-AVERAGED gradient and scaled
noise by 1/batch_size, then estimated epsilon with a hand-rolled composition
formula. That is not per-sample DP-SGD and the epsilon numbers it produced were
not a valid privacy guarantee. This version uses Opacus for true per-sample
gradient clipping + calibrated Gaussian noise + an RDP accountant.

Two DP scopes:
  --dp-scope head  Freeze the pretrained backbone, DP-SGD only the classifier
                    head. Far fewer per-sample gradients to compute -> fits in
                    4GB VRAM, and empirically recovers most of the utility lost
                    to full-network DP-SGD on a small dataset.
  --dp-scope full   DP-SGD on the whole network (BatchNorm -> GroupNorm via
                    Opacus's ModuleValidator, since BatchNorm mixes information
                    across the batch and is incompatible with per-sample DP).
                    Memory-heavy — prefer Colab (16GB) over the local 4GB GPU.
"""

import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centralised_train import (
    MRIDataset,
    compute_class_weights,
    compute_metrics,
    evaluate,
    precision_recall_fscore_support,
)
from models import get_model
from splits import load_split


def build_dp_model_and_optimizer(model_name, dp_scope, lr, weight_decay):
    """Construct a model prepared for the given DP scope (freeze / GroupNorm-fix)."""
    from opacus.validators import ModuleValidator

    freeze_backbone = (dp_scope == 'head')
    model = get_model(model_name, num_classes=3, pretrained=True, freeze_backbone=freeze_backbone)

    if dp_scope == 'full' and not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=weight_decay)
    return model, optimizer


def train_with_dp(model_name, train_loader, val_loader, criterion, device,
                  epochs, lr, weight_decay, dp_scope, target_epsilon, target_delta,
                  max_grad_norm, checkpoint_path=None, resume=False):
    """Real per-sample DP-SGD training loop with epoch-level checkpoint/resume.

    Noise multiplier is calibrated ONCE from (target_epsilon, target_delta, sample_rate,
    epochs) so that resuming doesn't silently recalibrate against a shorter remaining
    horizon (which would change the achieved epsilon at the end).
    """
    from opacus import PrivacyEngine
    from opacus.accountants.utils import get_noise_multiplier

    model, optimizer = build_dp_model_and_optimizer(model_name, dp_scope, lr, weight_decay)
    model = model.to(device)

    sample_rate = train_loader.batch_size / len(train_loader.dataset)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon, target_delta=target_delta,
        sample_rate=sample_rate, epochs=epochs, accountant='rdp',
    )

    pe = PrivacyEngine(accountant='rdp')
    dp_model, dp_optimizer, dp_loader = pe.make_private(
        module=model, optimizer=optimizer, data_loader=train_loader,
        noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
        # 'hooks' (Opacus default) breaks on ResNet's `out += identity` residual
        # add once the backbone is trainable — RuntimeError about modifying a
        # view in-place inside a custom Function's backward. 'ew' (ExpandedWeights)
        # computes per-sample grads without hooking into submodule forward/backward,
        # so it's unaffected. Only needed for dp_scope='full'; head-scope keeps the
        # already-validated default since the frozen backbone never exercises this.
        grad_sample_mode='ew' if dp_scope == 'full' else 'hooks',
    )

    # ExpandedWeights ('ew') mode ties per-sample-grad bookkeeping to every forward
    # pass on dp_model, not just backward — running validation directly through
    # dp_model trips "Current Expanded Weights accumulates the gradients... clear
    # grad_sample" because no backward/optimizer.step() follows the eval forward.
    # Evaluate on a separate plain (non-Opacus-wrapped) model, weight-synced from
    # dp_model each epoch, instead of fighting ExpandedWeights' forward-pass hooks.
    eval_model, _ = build_dp_model_and_optimizer(model_name, dp_scope, lr, weight_decay)
    eval_model = eval_model.to(device)

    def sync_eval_model():
        clean_state = {k.replace('_module.', ''): v for k, v in dp_model.state_dict().items()}
        eval_model.load_state_dict(clean_state)
        return eval_model

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'epsilon': []}
    best_val_f1 = 0.0
    best_state = None
    best_epoch = 0
    start_epoch = 1

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        dp_model.load_state_dict(ckpt['model_state_dict'])
        dp_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        pe.accountant.history = ckpt['accountant_history']  # see module docstring
        history = ckpt['history']
        best_val_f1 = ckpt['best_val_f1']
        best_epoch = ckpt['best_epoch']
        best_state = ckpt.get('best_state')
        start_epoch = ckpt['epoch'] + 1
        resumed_epoch = ckpt['epoch']
        # ckpt was loaded with map_location=device — its raw tensors (a full
        # duplicate of model+optimizer state) would otherwise sit on the GPU for
        # the rest of training. Measured on VGG19: this caused a ~2.7x per-batch
        # slowdown from memory pressure (not an outright OOM) on a 4GB GPU.
        del ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  >>> Resumed DP training from epoch {resumed_epoch} "
              f"(best F1 so far: {best_val_f1:.3f}, ε so far: {pe.get_epsilon(target_delta):.2f})")

    print(f"  σ={noise_multiplier:.4f} (calibrated for target ε={target_epsilon}, "
          f"δ={target_delta}, {epochs} epochs, sample_rate={sample_rate:.4f})")

    for epoch in range(start_epoch, epochs + 1):
        dp_model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(dp_loader, desc=f"  epoch {epoch}/{epochs}", leave=False):
            if images.size(0) == 0:
                # Opacus's Poisson-sampled DPDataLoader independently includes each
                # sample with probability = sample_rate, so an empty batch is a real
                # (if rare) outcome, not a bug — Opacus's own per-sample-grad code
                # path raises on batch_size=0 rather than special-casing it, so the
                # caller (here) has to skip it.
                continue
            images, labels = images.to(device), labels.to(device)
            dp_optimizer.zero_grad()
            outputs = dp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            dp_optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        epsilon = pe.get_epsilon(target_delta)

        val_loss, val_acc, val_preds, val_labels_arr, val_probs = evaluate(sync_eval_model(), val_loader, criterion, device)
        _, _, val_f1, _ = precision_recall_fscore_support(val_labels_arr, val_preds, average='macro', zero_division=0)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epsilon'].append(epsilon)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.clone().cpu() for k, v in dp_model.state_dict().items()}

        print(f"  epoch {epoch:3d}/{epochs} | loss {train_loss:.4f} acc {train_acc:.1f}% | "
              f"val_acc {val_acc:.1f}% val_f1 {val_f1:.3f} | ε={epsilon:.2f}"
              f"{'  ★' if epoch == best_epoch else ''}")

        if checkpoint_path:
            tmp_path = checkpoint_path + '.tmp'
            torch.save({
                'epoch': epoch,
                'model_state_dict': dp_model.state_dict(),
                'optimizer_state_dict': dp_optimizer.state_dict(),
                'accountant_history': pe.accountant.history,
                'history': history,
                'best_val_f1': best_val_f1,
                'best_epoch': best_epoch,
                'best_state': best_state,
                'noise_multiplier': noise_multiplier,
            }, tmp_path)
            os.replace(tmp_path, checkpoint_path)

    if best_state is not None:
        dp_model.load_state_dict(best_state)

    final_epsilon = pe.get_epsilon(target_delta)
    # Strip Opacus's GradSampleModule wrapping ('_module.' prefix) so the returned
    # state_dict loads into a plain model with get_model(...).load_state_dict(...).
    clean_state = {k.replace('_module.', ''): v for k, v in dp_model.state_dict().items()}
    plain_model, _ = build_dp_model_and_optimizer(model_name, dp_scope, lr, weight_decay)
    plain_model.load_state_dict(clean_state)

    return plain_model, history, final_epsilon, noise_multiplier, best_epoch


def run_privacy_utility_experiment(data_dir, splits_path, results_dir, model_name='resnet50',
                                   dp_scope='head', epsilon_values=None, epochs=10,
                                   fold=0, seed=42, batch_size=16, target_delta=1e-3,
                                   resume=False):
    """Sweep epsilon at fixed fold/seed, always saving a checkpoint per epsilon cell
    (the original version never persisted DP checkpoints, which made downstream
    MIA/XAI-under-DP evaluation impossible without re-training)."""
    if epsilon_values is None:
        epsilon_values = [2.0, 5.0, 10.0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoints_dir = os.path.join(results_dir, 'checkpoints')
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    images = np.load(os.path.join(data_dir, 'all_images.npy'))
    labels = np.load(os.path.join(data_dir, 'all_labels.npy'))
    manifest_path = os.path.join(data_dir, 'manifest.csv')
    split = load_split(fold, manifest_path, splits_path)
    train_idx, val_idx, test_idx = split['train_idx'], split['val_idx'], split['test_idx']

    class_weights = compute_class_weights(labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_dataset = MRIDataset(images[train_idx], labels[train_idx], augment=True)
    val_dataset = MRIDataset(images[val_idx], labels[val_idx])
    test_dataset = MRIDataset(images[test_idx], labels[test_idx])
    # drop_last: DP-SGD's Poisson sampling assumes a fixed sample_rate = batch/N.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {}

    for target_eps in epsilon_values:
        print(f"\n{'='*60}\nDP-{dp_scope} experiment: target ε={target_eps}\n{'='*60}")
        run_tag = f'{model_name}_dp{dp_scope}_eps{target_eps}_f{fold}_s{seed}'
        checkpoint_path = os.path.join(checkpoints_dir, f'{run_tag}_train.pth')

        try:
            model, history, actual_eps, sigma, best_epoch = train_with_dp(
                model_name, train_loader, val_loader, criterion, device,
                epochs=epochs, lr=1e-4, weight_decay=1e-4, dp_scope=dp_scope,
                target_epsilon=target_eps, target_delta=target_delta,
                max_grad_norm=1.0, checkpoint_path=checkpoint_path, resume=resume,
            )
            model = model.to(device)
            test_loss, test_acc, test_preds, test_labels_arr, test_probs = evaluate(model, test_loader, criterion, device)
            metrics = compute_metrics(test_labels_arr, test_preds, test_probs)

            final_checkpoint_path = os.path.join(checkpoints_dir, f'{run_tag}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model_name, 'dp_scope': dp_scope,
                'target_epsilon': target_eps, 'actual_epsilon': actual_eps,
                'target_delta': target_delta, 'noise_multiplier': sigma,
                'fold': fold, 'seed': seed, 'best_epoch': best_epoch,
            }, final_checkpoint_path)

            run_meta = {
                'run_tag': run_tag, 'model': model_name, 'fold': fold, 'seed': seed,
                'train_subjects': split['train_subjects'], 'val_subjects': split['val_subjects'],
                'test_subjects': split['test_subjects'],
                'dp_scope': dp_scope, 'target_epsilon': target_eps, 'actual_epsilon': actual_eps,
            }
            with open(os.path.join(metrics_dir, f'{run_tag}_run_meta.json'), 'w') as f:
                json.dump(run_meta, f, indent=2)
            with open(os.path.join(metrics_dir, f'{run_tag}_history.json'), 'w') as f:
                json.dump(history, f, indent=2)

            results[str(target_eps)] = {
                'target_epsilon': target_eps,
                'actual_epsilon': actual_eps,
                'noise_multiplier': sigma,
                'accuracy': float(metrics['accuracy']),
                'f1_macro': float(metrics['f1_macro']),
                'auroc_macro': float(metrics['auroc_macro']),
                'precision_macro': float(metrics['precision_macro']),
                'recall_macro': float(metrics['recall_macro']),
                'checkpoint': final_checkpoint_path,
            }
            print(f"  Result: ε={actual_eps:.2f}, Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

        except Exception as e:
            print(f"  [ERROR] ε={target_eps}: {e}")
            import traceback
            traceback.print_exc()
            results[str(target_eps)] = {'error': str(e)}

        torch.cuda.empty_cache()

    # No-DP reference (ε = None, not the literal float('inf') that broke JSON before)
    print(f"\n{'='*60}\nNo-DP reference ({dp_scope}-equivalent architecture)\n{'='*60}")
    ref_model, ref_optimizer = build_dp_model_and_optimizer(model_name, dp_scope, 1e-4, 1e-4)
    ref_model = ref_model.to(device)
    from centralised_train import train_one_epoch
    for _ep in tqdm(range(1, epochs + 1), desc="no-DP reference"):
        train_one_epoch(ref_model, train_loader, criterion, ref_optimizer, device)
    test_loss, test_acc, test_preds, test_labels_arr, test_probs = evaluate(ref_model, test_loader, criterion, device)
    no_dp_metrics = compute_metrics(test_labels_arr, test_preds, test_probs)
    no_dp_tag = f'{model_name}_dp{dp_scope}_epsNone_f{fold}_s{seed}'
    torch.save({'model_state_dict': ref_model.state_dict(), 'model_name': model_name,
               'dp_scope': dp_scope, 'fold': fold, 'seed': seed},
              os.path.join(checkpoints_dir, f'{no_dp_tag}.pth'))
    results['no_dp'] = {
        'target_epsilon': None, 'actual_epsilon': None,
        'accuracy': float(no_dp_metrics['accuracy']), 'f1_macro': float(no_dp_metrics['f1_macro']),
        'auroc_macro': float(no_dp_metrics['auroc_macro']),
    }
    torch.cuda.empty_cache()

    results_path = os.path.join(metrics_dir, f'{model_name}_dp{dp_scope}_f{fold}_s{seed}_privacy_utility.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)  # no float('inf') anywhere -> valid JSON
    print(f"\n  ✓ Results saved: {results_path}")

    plot_privacy_utility_curve(results, os.path.join(results_dir, 'figures'), f'{model_name}_dp{dp_scope}')
    return results


def plot_privacy_utility_curve(results, save_dir, tag):
    """Plot accuracy vs ε Pareto frontier."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epsilons, accuracies, f1_scores = [], [], []
    no_dp_acc = None
    for eps_key, metrics in results.items():
        if 'error' in metrics:
            continue
        if eps_key == 'no_dp':
            no_dp_acc = metrics['accuracy'] * 100
            continue
        epsilons.append(metrics['actual_epsilon'])
        accuracies.append(metrics['accuracy'] * 100)
        f1_scores.append(metrics.get('f1_macro', 0) * 100)

    order = np.argsort(epsilons)
    epsilons = np.array(epsilons)[order]
    accuracies = np.array(accuracies)[order]
    f1_scores = np.array(f1_scores)[order]

    ax.plot(epsilons, accuracies, 'bo-', linewidth=2, markersize=8, label='Accuracy')
    ax.plot(epsilons, f1_scores, 'rs--', linewidth=2, markersize=8, label='F1-Score')
    if no_dp_acc is not None:
        ax.axhline(no_dp_acc, color='gray', linestyle=':', label=f'No-DP reference ({no_dp_acc:.1f}%)')

    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Performance (%)', fontsize=12)
    ax.set_title(f'Privacy–Utility Trade-off ({tag})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{tag}_privacy_utility_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Privacy-utility curve saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='PPXFL DP Training (Opacus per-sample DP-SGD)')
    parser.add_argument('--model', type=str, default='resnet50', choices=['vgg19', 'resnet50'])
    parser.add_argument('--dp-scope', type=str, default='head', choices=['head', 'full'])
    parser.add_argument('--mode', type=str, default='experiment', choices=['single', 'experiment'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Small default — per-sample gradients are memory-heavy, esp. dp-scope full')
    parser.add_argument('--target-epsilon', type=float, default=2.0)
    parser.add_argument('--target-delta', type=float, default=1e-3)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--splits-path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--tag-suffix', type=str, default='')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'processed')
    if args.splits_path is None:
        args.splits_path = os.path.join(project_root, 'data', 'splits', 'splits_v1.json')
    results_dir = args.results_dir or os.path.join(project_root, 'results')

    if args.mode == 'experiment':
        run_privacy_utility_experiment(
            args.data_dir, args.splits_path, results_dir, args.model,
            dp_scope=args.dp_scope, epsilon_values=[2.0, 5.0, 10.0],
            epochs=args.epochs, fold=args.fold, seed=args.seed,
            batch_size=args.batch_size, target_delta=args.target_delta,
            resume=args.resume,
        )
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = np.load(os.path.join(args.data_dir, 'all_images.npy'))
        labels = np.load(os.path.join(args.data_dir, 'all_labels.npy'))
        manifest_path = os.path.join(args.data_dir, 'manifest.csv')
        split = load_split(args.fold, manifest_path, args.splits_path)

        train_dataset = MRIDataset(images[split['train_idx']], labels[split['train_idx']], augment=True)
        val_dataset = MRIDataset(images[split['val_idx']], labels[split['val_idx']])
        test_dataset = MRIDataset(images[split['test_idx']], labels[split['test_idx']])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        class_weights = compute_class_weights(labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        checkpoints_dir = os.path.join(results_dir, 'checkpoints')
        metrics_dir = os.path.join(results_dir, 'metrics')
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        run_tag = f'{args.model}_dp{args.dp_scope}_eps{args.target_epsilon}_f{args.fold}_s{args.seed}{args.tag_suffix}'
        checkpoint_path = os.path.join(checkpoints_dir, f'{run_tag}_train.pth')

        model, history, epsilon, sigma, best_epoch = train_with_dp(
            args.model, train_loader, val_loader, criterion, device,
            epochs=args.epochs, lr=1e-4, weight_decay=1e-4, dp_scope=args.dp_scope,
            target_epsilon=args.target_epsilon, target_delta=args.target_delta,
            max_grad_norm=args.max_grad_norm, checkpoint_path=checkpoint_path, resume=args.resume,
        )

        print(f"\nFinal ε: {epsilon:.2f}")
        torch.save({'model_state_dict': model.state_dict(), 'model_name': args.model,
                   'dp_scope': args.dp_scope, 'actual_epsilon': epsilon, 'fold': args.fold, 'seed': args.seed},
                  os.path.join(checkpoints_dir, f'{run_tag}.pth'))

        # Held-out test evaluation + run_meta/history — previously only written by
        # --mode experiment; --mode single (what run_experiments.py's B5/B6 cells
        # actually use) saved the checkpoint only, silently skipping test-set
        # accuracy and the membership ground truth MIA needs. run_experiments.py's
        # done_check only checks the checkpoint file, so this went unnoticed.
        model = model.to(device)
        test_loss, test_acc, test_preds, test_labels_arr, test_probs = evaluate(model, test_loader, criterion, device)
        metrics = compute_metrics(test_labels_arr, test_preds, test_probs)
        print(f"  Test — Acc: {metrics['accuracy']*100:.1f}%  F1: {metrics['f1_macro']:.3f}  "
              f"AUROC: {metrics['auroc_macro']:.3f}")

        run_meta = {
            'run_tag': run_tag, 'model': args.model, 'fold': args.fold, 'seed': args.seed,
            'train_subjects': split['train_subjects'], 'val_subjects': split['val_subjects'],
            'test_subjects': split['test_subjects'],
            'dp_scope': args.dp_scope, 'target_epsilon': args.target_epsilon, 'actual_epsilon': epsilon,
        }
        with open(os.path.join(metrics_dir, f'{run_tag}_run_meta.json'), 'w') as f:
            json.dump(run_meta, f, indent=2)
        with open(os.path.join(metrics_dir, f'{run_tag}_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(metrics_dir, f'{run_tag}_metrics.json'), 'w') as f:
            json.dump({
                'run_tag': run_tag, 'model': args.model, 'fold': args.fold, 'seed': args.seed,
                'dp_scope': args.dp_scope, 'target_epsilon': args.target_epsilon, 'actual_epsilon': epsilon,
                'noise_multiplier': sigma, 'best_epoch': best_epoch,
                'accuracy': float(metrics['accuracy']), 'f1_macro': float(metrics['f1_macro']),
                'auroc_macro': float(metrics['auroc_macro']),
                'precision_macro': float(metrics['precision_macro']), 'recall_macro': float(metrics['recall_macro']),
            }, f, indent=2)


if __name__ == '__main__':
    main()
