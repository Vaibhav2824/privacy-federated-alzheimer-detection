"""
models.py — CNN Model Definitions
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Defines VGG19 and ResNet50 with 3-class classification heads.
Uses ImageNet pre-trained weights with modified final layers.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_vgg19(num_classes=3, pretrained=True, freeze_backbone=False):
    """
    VGG19 with modified classifier for 3-class Alzheimer's classification.
    
    Args:
        num_classes: Number of output classes (CN=0, MCI=1, AD=2)
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze feature extractor layers
    
    Returns:
        model: VGG19 model
    """
    weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg19(weights=weights)
    
    # Modify first conv layer to accept 1-channel grayscale input
    # Original: Conv2d(3, 64, ...)
    original_conv = model.features[0]
    model.features[0] = nn.Conv2d(
        1, 64, 
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding
    )
    
    # Initialise new conv layer with averaged pretrained weights
    if pretrained:
        with torch.no_grad():
            model.features[0].weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
    
    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes),
    )
    
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    
    return model


def get_resnet50(num_classes=3, pretrained=True, freeze_backbone=False):
    """
    ResNet50 with modified FC layer for 3-class Alzheimer's classification.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze feature extractor layers
    
    Returns:
        model: ResNet50 model
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Modify first conv layer for 1-channel input
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(
        1, 64,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )
    
    if pretrained:
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
    
    # Replace final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'conv1' not in name:
                param.requires_grad = False
    
    return model


def get_model(model_name='vgg19', num_classes=3, pretrained=True, freeze_backbone=False):
    """Factory function to get model by name."""
    if model_name.lower() == 'vgg19':
        return get_vgg19(num_classes, pretrained, freeze_backbone)
    elif model_name.lower() == 'resnet50':
        return get_resnet50(num_classes, pretrained, freeze_backbone)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'vgg19' or 'resnet50'.")


def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Quick test
    for name in ['vgg19', 'resnet50']:
        model = get_model(name, num_classes=3)
        total, trainable = count_parameters(model)
        print(f"{name}: Total={total:,} | Trainable={trainable:,}")
        
        # Test forward pass
        dummy = torch.randn(2, 1, 224, 224)
        out = model(dummy)
        print(f"  Input: {dummy.shape} → Output: {out.shape}")
        print()
