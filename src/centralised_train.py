"""
centralised_train.py — Centralised Baseline Training
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Trains VGG19/ResNet50 on the full (non-partitioned) ADNI dataset as the
performance upper bound for comparison with federated variants.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model, count_parameters


class MRIDataset(Dataset):
    """PyTorch Dataset for preprocessed MRI slices."""
    
    def __init__(self, images, labels, transform=None, augment=False):
        """
        Args:
            images: numpy array of shape (N, H, W) 
            labels: numpy array of shape (N,) with class indices
            transform: optional torchvision transforms
            augment: apply random augmentation
        """
        self.images = torch.FloatTensor(images).unsqueeze(1)  # (N, 1, H, W)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-1])
            # Random Gaussian noise
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(image) * 0.02
                image = image + noise
                image = torch.clamp(image, 0, 1)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def compute_class_weights(labels):
    """Compute inverse frequency class weights for imbalanced data."""
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(classes) * counts)
    return torch.FloatTensor(weights)


def train_one_epoch(model, loader, criterion, optimiser, device):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model, return loss, accuracy, all predictions and labels."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    total = len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = 100.0 * np.sum(all_preds == all_labels) / total
    avg_loss = running_loss / total
    
    return avg_loss, accuracy, all_preds, all_labels, all_probs


def compute_metrics(y_true, y_pred, y_probs, class_names=['CN', 'MCI', 'AD']):
    """Compute comprehensive classification metrics."""
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class and macro metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2]
    )
    
    for i, name in enumerate(class_names):
        metrics[f'precision_{name}'] = precision[i]
        metrics[f'recall_{name}'] = recall[i]
        metrics[f'f1_{name}'] = f1[i]
    
    # Macro averages
    metrics['precision_macro'] = np.mean(precision)
    metrics['recall_macro'] = np.mean(recall)
    metrics['f1_macro'] = np.mean(f1)
    
    # AUROC (one-vs-rest)
    try:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        metrics['auroc_macro'] = roc_auc_score(y_bin, y_probs, multi_class='ovr', average='macro')
        for i, name in enumerate(class_names):
            metrics[f'auroc_{name}'] = roc_auc_score(y_bin[:, i], y_probs[:, i])
    except ValueError:
        metrics['auroc_macro'] = 0.0
    
    return metrics


def plot_training_curves(history, save_dir, model_name):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} — Training Curves', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training curves saved: {path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved: {save_path}")


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """Plot ROC curves for each class (one-vs-rest)."""
    from sklearn.metrics import roc_curve, auc
    
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    colors = ['#2196F3', '#FF9800', '#F44336']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curves saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='PPXFL Centralised Baseline Training')
    parser.add_argument('--model', type=str, default='vgg19', choices=['vgg19', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--freeze-epochs', type=int, default=5,
                        help='Epochs with frozen backbone (feature extraction)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'processed')
    
    results_dir = os.path.join(project_root, 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Load data
    print("\nLoading preprocessed data...")
    images = np.load(os.path.join(args.data_dir, 'all_images.npy'))
    labels = np.load(os.path.join(args.data_dir, 'all_labels.npy'))
    print(f"  Total: {len(images)} samples")
    print(f"  Classes: CN={np.sum(labels==0)}, MCI={np.sum(labels==1)}, AD={np.sum(labels==2)}")
    
    # Create dataset
    dataset = MRIDataset(images, labels)
    
    # 80/10/10 split
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create augmented training set wrapper
    train_images = images[list(train_set.indices)]
    train_labels = labels[list(train_set.indices)]
    train_dataset = MRIDataset(train_images, train_labels, augment=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    
    print(f"  Train: {len(train_dataset)} | Val: {len(val_set)} | Test: {len(test_set)}")
    
    # Compute class weights
    class_weights = compute_class_weights(labels).to(device)
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    
    # Initialise model
    print(f"\nInitialising {args.model}...")
    model = get_model(args.model, num_classes=3, pretrained=True, freeze_backbone=True)
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable (frozen backbone)")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_f1 = 0.0
    best_epoch = 0
    
    total_epochs = args.epochs
    
    print(f"\n{'='*60}")
    print(f"Starting training: {total_epochs} epochs")
    print(f"  Phase 1: {args.freeze_epochs} epochs (frozen backbone)")
    print(f"  Phase 2: {total_epochs - args.freeze_epochs} epochs (full fine-tuning)")
    print(f"{'='*60}\n")
    
    for epoch in range(1, total_epochs + 1):
        # Unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            print(f"\n>>> Unfreezing backbone at epoch {epoch}")
            for param in model.parameters():
                param.requires_grad = True
            # Reset optimiser with lower lr for backbone
            optimiser = optim.Adam([
                {'params': model.parameters(), 'lr': args.lr * 0.1}
            ], weight_decay=args.weight_decay)
            _, trainable_params = count_parameters(model)
            print(f"  Trainable parameters: {trainable_params:,}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimiser, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Compute val F1 for model selection
        _, _, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='macro', zero_division=0
        )
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            checkpoint_path = os.path.join(results_dir, f'best_{args.model}_centralised.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, checkpoint_path)
        
        phase = "frozen" if epoch <= args.freeze_epochs else "fine-tune"
        print(f"Epoch {epoch:3d}/{total_epochs} [{phase}] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% F1: {val_f1:.3f}"
              f"{'  ★' if epoch == best_epoch else ''}")
    
    print(f"\nBest epoch: {best_epoch} (Val F1: {best_val_f1:.3f})")
    
    # Load best model and evaluate on test set
    print(f"\nEvaluating best model on test set...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )
    
    # Compute comprehensive metrics
    class_names = ['CN', 'MCI', 'AD']
    metrics = compute_metrics(test_labels, test_preds, test_probs, class_names)
    
    print(f"\n{'='*60}")
    print(f"Test Results — {args.model} (Centralised)")
    print(f"{'='*60}")
    print(classification_report(test_labels, test_preds, target_names=class_names, digits=4))
    print(f"  AUROC (macro): {metrics['auroc_macro']:.4f}")
    
    # Save metrics
    metrics['model'] = args.model
    metrics['setting'] = 'centralised'
    metrics['best_epoch'] = best_epoch
    metrics['total_epochs'] = total_epochs
    
    metrics_path = os.path.join(metrics_dir, f'{args.model}_centralised_metrics.json')
    # Convert numpy types because json does not handle them
    serialisable = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"  ✓ Metrics saved: {metrics_path}")
    
    # Generate plots
    plot_training_curves(history, figures_dir, f'{args.model}_centralised')
    plot_confusion_matrix(test_labels, test_preds, class_names,
                         os.path.join(figures_dir, f'{args.model}_centralised_cm.png'))
    plot_roc_curves(test_labels, test_probs, class_names,
                   os.path.join(figures_dir, f'{args.model}_centralised_roc.png'))
    
    # Save training history
    history_path = os.path.join(metrics_dir, f'{args.model}_centralised_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Centralised {args.model} training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
