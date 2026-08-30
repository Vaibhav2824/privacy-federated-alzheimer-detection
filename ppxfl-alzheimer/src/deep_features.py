"""
deep_features.py — frozen-backbone embeddings of the MNI-standardised slices.

A logistic head over ResNet50's 2048-dimensional pooled features has
``2048 x 3 + 3 = 6,147`` parameters, which is exactly the head-scope
configuration the previous version of this study perturbed.  Writing those
embeddings out in the same format as the region features means the federated
and differential-privacy machinery runs over them unchanged, so the privacy
analysis gets a *measured* point at 6,147 parameters alongside the measured
point at 423 and the analytic point at 23.5M — three dimensions, one mechanism,
one cohort.

The backbone is frozen and evaluated once, so this is a feature extraction pass
rather than training: a subject's embedding is the mean of its slice
embeddings, matching the subject-level aggregation used everywhere else.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def extract(stacks, backbone: str = "resnet50", batch_size: int = 32,
            device: str | None = None, progress: bool = True) -> np.ndarray:
    """Mean pooled-feature embedding per subject.

    ``stacks`` is ``(n_subjects, n_levels, 3, H, W)``.  Returns
    ``(n_subjects, n_features)``.
    """
    import torch
    import torch.nn as nn

    from .cnn_v3 import get_backbone

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_backbone(backbone, num_classes=3, pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Identity()
    model = model.to(device).eval()

    n_subjects, n_levels = stacks.shape[0], stacks.shape[1]
    flat = stacks.reshape(n_subjects * n_levels, *stacks.shape[2:])
    out = np.zeros((n_subjects * n_levels, n_features), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            batch = torch.from_numpy(
                flat[start:start + batch_size].astype(np.float32)
            ).to(device)
            out[start:start + batch_size] = model(batch).cpu().numpy()
            if progress and start % (batch_size * 20) == 0:
                print(f"  embedded {start}/{flat.shape[0]} slices", flush=True)

    return out.reshape(n_subjects, n_levels, n_features).mean(axis=1)


def main(argv=None) -> int:
    from .features import save_features
    from .slices_v3 import main as _  # noqa: F401  (documents the producer)

    parser = argparse.ArgumentParser(description="frozen-backbone slice embeddings")
    parser.add_argument("--slices", default=os.path.join("data", "mni2mm", "slices.npz"))
    parser.add_argument("--out",
                        default=os.path.join("data", "mni2mm", "deep_features.npz"))
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    data = np.load(args.slices, allow_pickle=True)
    stacks = data["X"]
    print(f"{stacks.shape[0]} subjects x {stacks.shape[1]} MNI levels")

    embeddings = extract(stacks, backbone=args.backbone, batch_size=args.batch_size)
    names = [f"{args.backbone}::f{i}" for i in range(embeddings.shape[1])]
    save_features(args.out, embeddings, data["y"].astype(int),
                  list(data["subjects"]), list(data["sites"]), names)
    dimension = embeddings.shape[1] * 3 + 3
    print(f"wrote {args.out} with shape {embeddings.shape} "
          f"(logistic head dimension {dimension:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
