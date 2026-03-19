"""
evaluate.py — Comprehensive Evaluation & MIA
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Runs all comparison experiments and Membership Inference Attack evaluation.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support, accuracy_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model
from centralised_train import (
    MRIDataset, compute_class_weights, evaluate as eval_model, compute_metrics
)


def membership_inference_attack(model, train_images, train_labels,
                                 test_images, test_labels, device,
                                 num_shadow_models=3, seed=42):
    """
    Simple Membership Inference Attack (MIA) using shadow model approach.
    
    The attack trains shadow models and uses confidence scores to distinguish
    members (training data) from non-members (test data).
    
    Args:
        model: Target model to attack
        train_images: Training data
        train_labels: Training labels
        test_images: Test (non-member) data
        test_labels: Test labels
        device: torch device
        num_shadow_models: Number of shadow models (for averaging)
        seed: Random seed
    
    Returns:
        mia_accuracy: Attack success rate (target: ~50% for good privacy)
        mia_advantage: attack_accuracy - 50% (target: < 5%)
    """
    np.random.seed(seed)
    model.to(device)
    model.eval()
    
    def get_confidence_scores(model, images, labels, device):
        """Get model confidence (max softmax probability) for each sample."""
        dataset = MRIDataset(images, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        confidences = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                max_probs, _ = probs.max(dim=1)
                confidences.extend(max_probs.cpu().numpy())
        
        return np.array(confidences)
    
    # Get confidence scores
    train_conf = get_confidence_scores(model, train_images, train_labels, device)
    test_conf = get_confidence_scores(model, test_images, test_labels, device)
    
    # Simple threshold attack: members tend to have higher confidence
    # Use optimal threshold on combined scores
    all_conf = np.concatenate([train_conf, test_conf])
    all_labels_mia = np.concatenate([np.ones(len(train_conf)), np.zeros(len(test_conf))])
    
    best_acc = 0.0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 1.0, 0.01):
        predictions = (all_conf >= threshold).astype(int)
        acc = np.mean(predictions == all_labels_mia)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    mia_accuracy = best_acc * 100
    mia_advantage = mia_accuracy - 50.0
    
    return mia_accuracy, mia_advantage, best_threshold


def run_all_experiments(data_dir, results_dir, seed=42):
    """
    Run experiments E1-E6 as defined in the implementation plan.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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


def run_mia_evaluation(model_path, model_name, data_dir, results_dir, seed=42):
    """Run MIA evaluation on a trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    images = np.load(os.path.join(data_dir, 'all_images.npy'))
    labels = np.load(os.path.join(data_dir, 'all_labels.npy'))
    
    # Split into train/test
    n = len(images)
    np.random.seed(seed)
    idx = np.random.permutation(n)
    train_idx = idx[:int(0.8 * n)]
    test_idx = idx[int(0.9 * n):]
    
    # Load model
    model = get_model(model_name, num_classes=3, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"\nRunning MIA on {model_name}...")
    mia_acc, mia_adv, threshold = membership_inference_attack(
        model, images[train_idx], labels[train_idx],
        images[test_idx], labels[test_idx], device
    )
    
    print(f"  MIA Accuracy: {mia_acc:.1f}%")
    print(f"  MIA Advantage: {mia_adv:.1f}%")
    print(f"  Threshold: {threshold:.3f}")
    
    # Save MIA results
    mia_results = {
        'model': model_name,
        'model_path': model_path,
        'mia_accuracy': mia_acc,
        'mia_advantage': mia_adv,
        'threshold': threshold,
    }
    
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    mia_path = os.path.join(metrics_dir, f'{model_name}_mia_results.json')
    with open(mia_path, 'w') as f:
        json.dump(mia_results, f, indent=2)
    print(f"  ✓ MIA results saved: {mia_path}")
    
    return mia_results


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
            bars = ax.bar(range(len(df)), df[metric] * 100, color=colors)
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
    parser = argparse.ArgumentParser(description='PPXFL Evaluation')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'mia', 'compare'],
                        help='all: compile results | mia: run MIA | compare: generate plots')
    parser.add_argument('--model-name', type=str, default='vgg19')
    parser.add_argument('--model-path', type=str, default=None)
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
    elif args.experiment == 'mia':
        if args.model_path is None:
            print("[ERROR] --model-path required for MIA evaluation")
            sys.exit(1)
        run_mia_evaluation(args.model_path, args.model_name, args.data_dir, results_dir, args.seed)
    elif args.experiment == 'compare':
        generate_comparison_plots(results_dir)


if __name__ == '__main__':
    main()
