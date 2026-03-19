"""
gradcam_analysis.py — Grad-CAM Explainability Analysis
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Generates Grad-CAM heatmaps showing which MRI regions drive classification decisions.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model
from centralised_train import MRIDataset


class GradCAM:
    """Grad-CAM implementation for CNN models."""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: The convolutional layer to compute Grad-CAM on
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = predicted class)
        
        Returns:
            heatmap: numpy array (H, W) in [0, 1]
            predicted_class: The class used for Grad-CAM
            confidence: Softmax probability for the target class
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # Global average pooling
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions
        
        # Resize to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class, confidence


def get_target_layer(model, model_name):
    """Get the last convolutional layer for Grad-CAM."""
    if model_name == 'vgg19':
        return model.features[-1]  # Last conv layer in VGG19
    elif model_name == 'resnet50':
        return model.layer4[-1].conv3  # Last conv in ResNet50
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_heatmap_overlay(image, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on the original MRI image."""
    # Convert grayscale image to RGB
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image
    
    # Apply jet colormap to heatmap
    colormap = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Blend
    overlay = (1 - alpha) * image_rgb + alpha * colormap
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def generate_gradcam_grid(model, model_name, images, labels, device,
                          num_samples=10, save_path=None):
    """
    Generate Grad-CAM heatmaps for multiple samples across classes.
    
    Args:
        model: Trained model
        model_name: 'vgg19' or 'resnet50'
        images: numpy array of images (N, H, W)
        labels: numpy array of labels (N,)
        device: torch device
        num_samples: Samples per class
        save_path: Path to save the grid image
    """
    class_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)
    
    model.to(device)
    model.eval()
    
    # Get correctly classified samples per class
    correct_samples = {0: [], 1: [], 2: []}
    
    for i in range(len(images)):
        img_tensor = torch.FloatTensor(images[i]).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor).argmax(dim=1).item()
        if pred == labels[i]:
            correct_samples[labels[i]].append(i)
    
    # Create grid: 3 rows (classes) × num_samples*2 cols (original + heatmap)
    n_per_class = min(num_samples, min(len(v) for v in correct_samples.values()))
    
    if n_per_class == 0:
        print("  [WARNING] Not enough correctly classified samples for Grad-CAM.")
        return
    
    fig, axes = plt.subplots(3, n_per_class * 2, figsize=(4 * n_per_class, 12))
    fig.suptitle(f'Grad-CAM Heatmaps — {model_name}', fontsize=16, fontweight='bold')
    
    for row, class_id in enumerate([0, 1, 2]):
        sample_indices = correct_samples[class_id][:n_per_class]
        
        for col, idx in enumerate(sample_indices):
            img = images[idx]
            img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)
            img_tensor.requires_grad_(True)
            
            heatmap, pred_class, conf = grad_cam.generate(img_tensor, target_class=class_id)
            overlay = create_heatmap_overlay(img, heatmap)
            
            # Original image
            ax_orig = axes[row, col * 2] if n_per_class > 1 else axes[row]
            ax_orig.imshow(img, cmap='gray')
            ax_orig.set_title(f'{class_names[class_id]}', fontsize=10)
            ax_orig.axis('off')
            
            # Heatmap overlay
            ax_heat = axes[row, col * 2 + 1] if n_per_class > 1 else axes[row]
            ax_heat.imshow(overlay)
            ax_heat.set_title(f'Conf: {conf:.2f}', fontsize=10)
            ax_heat.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Grad-CAM grid saved: {save_path}")
    
    plt.close()


def generate_class_comparison(model, model_name, images, labels, device,
                               save_path=None):
    """Generate side-by-side Grad-CAM comparison across CN/MCI/AD."""
    class_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)
    model.to(device)
    model.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Grad-CAM Class Comparison — {model_name}', fontsize=16, fontweight='bold')
    
    for col, class_id in enumerate([0, 1, 2]):
        # Find a correctly classified sample
        for i in range(len(images)):
            if labels[i] == class_id:
                img_tensor = torch.FloatTensor(images[i]).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(img_tensor).argmax(dim=1).item()
                if pred == class_id:
                    img_tensor.requires_grad_(True)
                    heatmap, _, conf = grad_cam.generate(img_tensor, target_class=class_id)
                    overlay = create_heatmap_overlay(images[i], heatmap)
                    
                    axes[0, col].imshow(images[i], cmap='gray')
                    axes[0, col].set_title(f'{class_names[class_id]} — Original', fontsize=12)
                    axes[0, col].axis('off')
                    
                    axes[1, col].imshow(overlay)
                    axes[1, col].set_title(f'{class_names[class_id]} — Grad-CAM (conf={conf:.2f})', fontsize=12)
                    axes[1, col].axis('off')
                    break
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Class comparison saved: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PPXFL Grad-CAM Analysis')
    parser.add_argument('--model-name', type=str, default='vgg19', choices=['vgg19', 'resnet50'])
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'processed')
    
    xai_dir = os.path.join(project_root, 'results', 'xai')
    os.makedirs(xai_dir, exist_ok=True)
    
    # Load data
    images = np.load(os.path.join(args.data_dir, 'all_images.npy'))
    labels = np.load(os.path.join(args.data_dir, 'all_labels.npy'))
    
    # Load model
    model = get_model(args.model_name, num_classes=3, pretrained=False)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"\nGenerating Grad-CAM heatmaps for {args.model_name}...")
    
    # Generate heatmap grid
    generate_gradcam_grid(
        model, args.model_name, images, labels, device,
        num_samples=args.num_samples,
        save_path=os.path.join(xai_dir, f'{args.model_name}_gradcam_grid.png')
    )
    
    # Generate class comparison
    generate_class_comparison(
        model, args.model_name, images, labels, device,
        save_path=os.path.join(xai_dir, f'{args.model_name}_gradcam_comparison.png')
    )
    
    print("\nGrad-CAM analysis complete!")


if __name__ == '__main__':
    main()
