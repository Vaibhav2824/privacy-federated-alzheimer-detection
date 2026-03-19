"""
shap_analysis.py — SHAP Explainability Analysis
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Computes SHAP values for global feature importance using GradientExplainer.
GPU-friendly: uses small batches and limited samples for RTX 3050 Ti.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model
from centralised_train import MRIDataset


def compute_shap_values(model, images, labels, device, num_samples=50,
                        num_background=20, batch_size=10, seed=42):
    """
    Compute SHAP values using GradientExplainer.
    GPU-friendly: small batches, limited samples.

    Args:
        model: Trained PyTorch model
        images: numpy array (N, H, W)
        labels: numpy array (N,)
        device: torch device
        num_samples: Number of test samples to explain
        num_background: Number of background samples
        batch_size: Batch size for SHAP computation
        seed: Random seed

    Returns:
        shap_values: List of SHAP value arrays per class
        test_images: Images that were explained
        test_labels: Labels of explained images
    """
    import shap

    np.random.seed(seed)
    model.to(device)
    model.eval()

    # Prepare background data (small for GPU memory)
    bg_idx = np.random.choice(len(images), size=min(num_background, len(images)), replace=False)
    background = torch.FloatTensor(images[bg_idx]).unsqueeze(1).to(device)

    # Prepare test samples
    remaining_idx = np.setdiff1d(np.arange(len(images)), bg_idx)
    test_idx = np.random.choice(remaining_idx, size=min(num_samples, len(remaining_idx)), replace=False)
    test_images = images[test_idx]
    test_labels = labels[test_idx]

    print(f"\n  Computing SHAP values for {len(test_idx)} samples "
          f"(background: {len(bg_idx)} samples)...")

    # Use GradientExplainer (fast for neural networks)
    explainer = shap.GradientExplainer(model, background)

    # Compute SHAP values in batches to avoid GPU OOM
    all_shap = []
    num_batches = (len(test_images) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="  SHAP batches", unit="batch",
                  bar_format='{l_bar}{bar:30}{r_bar}'):
        start = i * batch_size
        end = min(start + batch_size, len(test_images))
        batch_tensor = torch.FloatTensor(test_images[start:end]).unsqueeze(1).to(device)

        batch_shap = explainer.shap_values(batch_tensor)
        all_shap.append(batch_shap)

        # Free GPU
        del batch_tensor
        torch.cuda.empty_cache()

    # Concatenate batch results
    # shap_values is list of arrays (one per class)
    if isinstance(all_shap[0], list):
        num_classes = len(all_shap[0])
        shap_values = []
        for c in range(num_classes):
            class_shap = np.concatenate([b[c] for b in all_shap], axis=0)
            shap_values.append(class_shap)
    else:
        shap_values = np.concatenate(all_shap, axis=0)

    # Free background from GPU
    del background
    torch.cuda.empty_cache()

    return shap_values, test_images, test_labels


def plot_shap_summary(shap_values, test_images, test_labels, save_dir, model_name):
    """Generate SHAP summary plots."""
    class_names = ['CN', 'MCI', 'AD']

    # Average absolute SHAP values per class
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'SHAP Feature Importance by Class — {model_name}',
                 fontsize=14, fontweight='bold')

    for i, (class_name, ax) in enumerate(zip(class_names, axes)):
        if isinstance(shap_values, list) and i < len(shap_values):
            sv = np.array(shap_values[i])
        else:
            sv = np.array(shap_values)

        # Average absolute SHAP values across samples → spatial importance map
        if len(sv.shape) == 4:  # (N, C, H, W)
            importance = np.abs(sv).mean(axis=(0, 1))
        elif len(sv.shape) == 3:  # (N, H, W)
            importance = np.abs(sv).mean(axis=0)
        else:
            # Flatten and try to reshape to 2D
            total_px = sv.shape[-1] if len(sv.shape) > 0 else 224*224
            side = int(np.sqrt(total_px))
            importance = np.abs(sv).mean(axis=0)
            if importance.ndim > 2:
                importance = importance.mean(axis=tuple(range(importance.ndim - 2)))
            elif importance.ndim == 1 and len(importance) == 224*224:
                importance = importance.reshape(224, 224)
            elif importance.ndim == 1:
                importance = importance.reshape(side, -1)
        
        # Ensure 2D
        if importance.ndim > 2:
            importance = importance.mean(axis=tuple(range(importance.ndim - 2)))

        im = ax.imshow(importance, cmap='hot', aspect='auto')
        ax.set_title(f'{class_name}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_shap_summary.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SHAP summary saved: {save_path}")


def plot_shap_examples(shap_values, test_images, test_labels, save_dir, model_name):
    """Plot SHAP overlays for individual examples."""
    class_names = {0: 'CN', 1: 'MCI', 2: 'AD'}

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'SHAP Explanations — Individual Samples ({model_name})',
                 fontsize=14, fontweight='bold')

    for row, class_id in enumerate([0, 1, 2]):
        class_mask = test_labels == class_id
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            continue

        idx = class_indices[0]
        img = test_images[idx]

        # Get SHAP values for this sample
        if isinstance(shap_values, list) and class_id < len(shap_values):
            sv = np.array(shap_values[class_id])
            local_idx = np.where(class_indices == idx)[0]
            sv_idx = local_idx[0] if len(local_idx) > 0 else 0
            shap_map = sv[sv_idx]
        else:
            sv = np.array(shap_values)
            shap_map = sv[idx]

        # Reduce to 2D (224, 224) regardless of input shape
        while shap_map.ndim > 2:
            # Reduce the first non-spatial axis by mean
            shap_map = shap_map.mean(axis=0)
        if shap_map.ndim == 1:
            side = int(np.sqrt(len(shap_map)))
            shap_map = shap_map.reshape(side, -1)

        # Original
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].set_title(f'{class_names[class_id]} — Original')
        axes[row, 0].axis('off')

        # Positive SHAP (drives prediction)
        pos_shap = np.maximum(shap_map, 0)
        axes[row, 1].imshow(pos_shap, cmap='Reds')
        axes[row, 1].set_title('Positive SHAP')
        axes[row, 1].axis('off')

        # Negative SHAP (opposing prediction)
        neg_shap = np.abs(np.minimum(shap_map, 0))
        axes[row, 2].imshow(neg_shap, cmap='Blues')
        axes[row, 2].set_title('Negative SHAP')
        axes[row, 2].axis('off')

        # Overlay — ensure shap_map and img have same spatial dimensions
        abs_shap = np.abs(shap_map)
        if abs_shap.max() > 0:
            abs_shap = abs_shap / abs_shap.max()
        # Resize abs_shap to match img if needed
        if abs_shap.shape != img.shape:
            from PIL import Image as PILImage
            abs_shap = np.array(PILImage.fromarray((abs_shap * 255).astype(np.uint8)).resize(
                (img.shape[1], img.shape[0]), PILImage.BILINEAR)) / 255.0
        overlay = np.stack([img] * 3, axis=-1)
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + abs_shap * 0.5, 0, 1)
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title('Overlay')
        axes[row, 3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_shap_examples.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SHAP examples saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='PPXFL SHAP Analysis')
    parser.add_argument('--model-name', type=str, default='resnet50', choices=['vgg19', 'resnet50'])
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples to explain (keep low for laptop GPU)')
    parser.add_argument('--background', type=int, default=20,
                        help='Number of background samples (keep low for laptop GPU)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for SHAP computation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'processed')

    xai_dir = os.path.join(project_root, 'results', 'xai')

    # Load data
    print("Loading data...")
    images = np.load(os.path.join(args.data_dir, 'all_images.npy'))
    labels = np.load(os.path.join(args.data_dir, 'all_labels.npy'))

    # Load model
    print(f"Loading {args.model_name} from {args.model_path}...")
    model = get_model(args.model_name, num_classes=3, pretrained=False)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"\nRunning SHAP analysis for {args.model_name}...")

    # Compute SHAP values
    shap_values, test_images, test_labels = compute_shap_values(
        model, images, labels, device,
        num_samples=args.samples,
        num_background=args.background,
        batch_size=args.batch_size
    )

    # Generate plots
    print("\nGenerating plots...")
    plot_shap_summary(shap_values, test_images, test_labels, xai_dir, args.model_name)
    plot_shap_examples(shap_values, test_images, test_labels, xai_dir, args.model_name)

    # Save SHAP values
    np.save(os.path.join(xai_dir, f'{args.model_name}_shap_values.npy'),
            np.array(shap_values, dtype=object), allow_pickle=True)

    print("\n✓ SHAP analysis complete!")


if __name__ == '__main__':
    main()
