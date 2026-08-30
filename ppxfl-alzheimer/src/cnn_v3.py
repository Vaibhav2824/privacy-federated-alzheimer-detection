"""
cnn_v3.py — ResNet on MNI-standardised 2.5D slices.

This is the same architecture family the earlier version of the study used, run
on the corrected input, so the two are comparable and the improvement can be
attributed to the representation rather than to a change of model.  It also
supplies the high-dimensional end of the privacy analysis: a trained 23.5M
parameter model whose subject-level DP behaviour can be measured rather than
only predicted from the noise-norm argument.

Two details matter for a fair comparison with the region-feature model:

* slices carry three adjacent parallel planes as three channels, so the
  ImageNet stem is used as trained instead of being averaged down to one
  channel;
* a subject's prediction is the mean of its slice probabilities, so the unit of
  evaluation is the subject in both models.
"""

from __future__ import annotations

import numpy as np


def get_backbone(name: str = "resnet18", num_classes: int = 3,
                 pretrained: bool = True, head_only: bool = False):
    """An ImageNet backbone with a fresh classification head.

    ``head_only`` freezes everything except the final linear layer, which is the
    small-dimension configuration used in the privacy comparison.  The freeze
    matches on an explicit ``fc.`` prefix: a substring test on ``"fc"`` also
    matches nothing here, but the equivalent test on ``"conv1"`` in the previous
    version left every bottleneck's ``conv1`` trainable and so left 4.3M
    parameters inside a mechanism that was supposed to cover 6,147.
    """
    import torch.nn as nn
    from torchvision import models

    builders = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    }
    if name not in builders:
        raise ValueError(f"unknown backbone: {name}")
    builder, weights = builders[name]
    model = builder(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if head_only:
        for parameter_name, parameter in model.named_parameters():
            parameter.requires_grad = parameter_name.startswith("fc.")
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                # Held fixed so running statistics stay outside anything the
                # privacy accounting has to cover.
                module.momentum = 0.0
                module.eval()
    return model


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class SliceDataset:
    """Slices of a subset of subjects, flattened to one sample per slice."""

    def __init__(self, stacks, labels, subject_index, augment: bool = False,
                 seed: int = 0):
        self.stacks = stacks
        self.labels = labels
        self.subject_index = np.asarray(subject_index)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        n_levels = stacks.shape[1]
        self.pairs = [
            (int(subject), level)
            for subject in self.subject_index
            for level in range(n_levels)
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item):
        import torch

        subject, level = self.pairs[item]
        image = self.stacks[subject, level].astype(np.float32)
        if self.augment:
            if self.rng.random() < 0.5:
                image = np.flip(image, axis=2).copy()
            shift = self.rng.integers(-6, 7, size=2)
            image = np.roll(image, tuple(shift), axis=(1, 2))
            image = image * float(self.rng.normal(1.0, 0.05))
        return torch.from_numpy(image), int(self.labels[subject]), subject


def train_one_model(stacks, labels, train_index, test_index, backbone: str = "resnet18",
                    head_only: bool = False, epochs: int = 12, batch_size: int = 32,
                    learning_rate: float = 3e-4, seed: int = 42, device: str | None = None):
    """Train on the training subjects and return subject-level test probabilities."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_backbone(backbone, num_classes=int(labels.max() + 1),
                         head_only=head_only).to(device)

    train_set = SliceDataset(stacks, labels, train_index, augment=True, seed=seed)
    test_set = SliceDataset(stacks, labels, test_index, augment=False, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=0)

    counts = np.bincount(labels[train_index], minlength=int(labels.max() + 1))
    weight = torch.tensor(
        (counts.sum() / np.maximum(counts, 1)) / len(counts), dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    for _ in range(epochs):
        model.train()
        if head_only:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimiser.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimiser.step()
        scheduler.step()

    model.eval()
    n_classes = int(labels.max() + 1)
    totals = np.zeros((stacks.shape[0], n_classes))
    seen = np.zeros(stacks.shape[0])
    with torch.no_grad():
        for images, _, subjects in test_loader:
            probability = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            for row, subject in zip(probability, subjects.numpy()):
                totals[subject] += row
                seen[subject] += 1
    keep = seen > 0
    totals[keep] /= seen[keep, None]
    return totals, model
