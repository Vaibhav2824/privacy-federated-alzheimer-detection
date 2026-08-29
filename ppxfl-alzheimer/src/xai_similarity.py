"""
xai_similarity.py — Explanation degradation under DP.

Core novelty artifact of the paper: quantifies how much Grad-CAM/SHAP
explanations shift, and how much less faithful they become, when a model is
trained with DP-SGD instead of standard SGD, as a function of epsilon.

All comparisons are computed on the SAME fold-0 test slices for both the DP
model and its non-DP reference — different image sets would confound
"explanation changed" with "different images". Restricting to held-out test
slices (not training slices, which the original gradcam_analysis.py sampled
from) also means the explanations being compared were never seen during
training by either model.

Two families of metric:
  - Similarity (DP heatmap vs non-DP reference heatmap, same input, same
    target class): SSIM, Spearman rank correlation, Dice overlap of the
    top-20%-by-value pixels.
  - Faithfulness (per model, standalone): deletion AUC (mask the
    most-important pixels first, measure how fast predicted-class probability
    drops — lower is more faithful) and insertion AUC (reveal the
    most-important pixels first on a blanked image, measure how fast
    probability rises — higher is more faithful).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradcam_analysis import GradCAM, get_target_layer
from models import get_model
from splits import load_split


def compute_gradcam_batch(model, model_name, images, labels, device):
    """Grad-CAM heatmap for every (image, true_label) pair — true label is used
    as the target class (not the predicted class) so DP and non-DP heatmaps
    are directly comparable even where the DP model's prediction differs."""
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)
    model.to(device)

    heatmaps = []
    for i in range(len(images)):
        img_tensor = torch.FloatTensor(images[i]).unsqueeze(0).unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)
        heatmap, _, _ = grad_cam.generate(img_tensor, target_class=int(labels[i]))
        heatmaps.append(heatmap)
    return np.stack(heatmaps)


def heatmap_similarity(heatmap_a, heatmap_b, top_frac=0.2):
    """SSIM, Spearman rho, Dice@top-k between two same-shape heatmaps in [0,1]."""
    ssim_val = ssim(heatmap_a, heatmap_b, data_range=1.0)
    rho, _ = spearmanr(heatmap_a.ravel(), heatmap_b.ravel())

    k = max(1, int(top_frac * heatmap_a.size))
    thresh_a = np.partition(heatmap_a.ravel(), -k)[-k]
    thresh_b = np.partition(heatmap_b.ravel(), -k)[-k]
    mask_a = heatmap_a >= thresh_a
    mask_b = heatmap_b >= thresh_b
    intersection = np.logical_and(mask_a, mask_b).sum()
    dice = 2.0 * intersection / (mask_a.sum() + mask_b.sum() + 1e-8)

    return {'ssim': float(ssim_val), 'spearman': float(rho), 'dice_top20': float(dice)}


def deletion_insertion_auc(model, image, heatmap, target_class, device, num_steps=20):
    """Deletion AUC (lower=more faithful) and insertion AUC (higher=more faithful)
    for one image, using the heatmap's pixel ranking as the masking order."""
    model.eval()
    h, w = heatmap.shape
    order = np.argsort(-heatmap.ravel())  # most important first
    total_pixels = h * w
    step_size = max(1, total_pixels // num_steps)

    base_image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
    baseline_value = float(image.mean())  # mean-fill, not zero — zero is off-distribution for normalised MRI

    def predict_prob(img_tensor):
        with torch.no_grad():
            probs = F.softmax(model(img_tensor), dim=1)
        return probs[0, target_class].item()

    deletion_probs, insertion_probs = [], []
    deleted = base_image.clone()
    inserted = torch.full_like(base_image, baseline_value)

    deletion_probs.append(predict_prob(deleted))
    insertion_probs.append(predict_prob(inserted))

    flat_deleted = deleted.view(-1)
    flat_inserted = inserted.view(-1)
    flat_original = base_image.view(-1)

    for step in range(num_steps):
        pixels = order[step * step_size: (step + 1) * step_size]
        flat_deleted[pixels] = baseline_value
        flat_inserted[pixels] = flat_original[pixels]
        deletion_probs.append(predict_prob(deleted))
        insertion_probs.append(predict_prob(inserted))

    x = np.linspace(0, 1, len(deletion_probs))
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # numpy>=2.0 renamed trapz -> trapezoid
    deletion_auc = float(trapezoid(deletion_probs, x))
    insertion_auc = float(trapezoid(insertion_probs, x))
    return deletion_auc, insertion_auc


def compute_shap_batch(model, images, background_images, device, batch_size=10):
    """SHAP GradientExplainer attributions for the target class at each slice."""
    import shap
    model.eval()
    background = torch.FloatTensor(background_images).unsqueeze(1).to(device)
    explainer = shap.GradientExplainer(model, background)

    all_maps = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        batch_tensor = torch.FloatTensor(batch).unsqueeze(1).to(device)
        shap_values = explainer.shap_values(batch_tensor)
        # shap_values: (batch, 1, H, W, num_classes) or list-of-arrays depending on shap version
        if isinstance(shap_values, list):
            per_class = np.stack(shap_values, axis=-1)  # (batch, 1, H, W, C)
        else:
            per_class = shap_values
        all_maps.append(per_class.squeeze(1))  # (batch, H, W, C)
    return np.concatenate(all_maps, axis=0)


def run_xai_similarity(dp_checkpoint, reference_checkpoint, model_name,
                       data_dir, splits_path, results_dir, fold=0, freeze_backbone_dp=False,
                       freeze_backbone_ref=False, dp_scope='none', max_slices=None,
                       include_shap=False, device=None):
    """Compare a DP checkpoint's explanations against a non-DP reference on the
    SAME fold test slices. Membership/eligibility not required here — we only
    need the test split, which both models share by construction (test subjects
    are never in any client / any training set for this fold)."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    manifest_path = os.path.join(data_dir, 'manifest.csv')
    split = load_split(fold, manifest_path, splits_path)
    images_all = np.load(os.path.join(data_dir, 'all_images.npy'))
    labels_all = np.load(os.path.join(data_dir, 'all_labels.npy'))
    test_idx = split['test_idx']
    if max_slices is not None and len(test_idx) > max_slices:
        rng = np.random.RandomState(42)
        test_idx = rng.choice(test_idx, size=max_slices, replace=False)
    images, labels = images_all[test_idx], labels_all[test_idx]

    def load_model(checkpoint_path, freeze_backbone, scope='none'):
        if scope == 'full':
            # Full-scope DP checkpoints have BatchNorm converted to GroupNorm
            # (Opacus ModuleValidator.fix, see dp_train.py) — plain get_model()
            # can't load their state_dict (same bug class as mia.py hit).
            from dp_train import build_dp_model_and_optimizer
            model, _ = build_dp_model_and_optimizer(model_name, 'full', lr=1e-4, weight_decay=1e-4)
        else:
            model = get_model(model_name, num_classes=3, pretrained=False, freeze_backbone=freeze_backbone)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        model = model.to(device)
        del ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    dp_model = load_model(dp_checkpoint, freeze_backbone_dp, scope=dp_scope)
    ref_model = load_model(reference_checkpoint, freeze_backbone_ref)

    print(f"Computing Grad-CAM for {len(images)} test slices (DP + reference)...")
    dp_heatmaps = compute_gradcam_batch(dp_model, model_name, images, labels, device)
    ref_heatmaps = compute_gradcam_batch(ref_model, model_name, images, labels, device)

    per_slice_similarity = [heatmap_similarity(dp_heatmaps[i], ref_heatmaps[i]) for i in range(len(images))]
    ssim_vals = [s['ssim'] for s in per_slice_similarity]
    spearman_vals = [s['spearman'] for s in per_slice_similarity]
    dice_vals = [s['dice_top20'] for s in per_slice_similarity]

    print("Computing deletion/insertion faithfulness (DP + reference)...")
    dp_del, dp_ins, ref_del, ref_ins = [], [], [], []
    for i in range(len(images)):
        d_del, d_ins = deletion_insertion_auc(dp_model, images[i], dp_heatmaps[i], int(labels[i]), device)
        r_del, r_ins = deletion_insertion_auc(ref_model, images[i], ref_heatmaps[i], int(labels[i]), device)
        dp_del.append(d_del)
        dp_ins.append(d_ins)
        ref_del.append(r_del)
        ref_ins.append(r_ins)

    results = {
        'dp_checkpoint': dp_checkpoint,
        'reference_checkpoint': reference_checkpoint,
        'model': model_name,
        'fold': fold,
        'n_slices': len(images),
        'gradcam_similarity': {
            'ssim_mean': float(np.mean(ssim_vals)), 'ssim_std': float(np.std(ssim_vals)),
            'spearman_mean': float(np.nanmean(spearman_vals)), 'spearman_std': float(np.nanstd(spearman_vals)),
            'dice_top20_mean': float(np.mean(dice_vals)), 'dice_top20_std': float(np.std(dice_vals)),
        },
        'faithfulness': {
            'dp_deletion_auc_mean': float(np.mean(dp_del)), 'dp_deletion_auc_std': float(np.std(dp_del)),
            'dp_insertion_auc_mean': float(np.mean(dp_ins)), 'dp_insertion_auc_std': float(np.std(dp_ins)),
            'ref_deletion_auc_mean': float(np.mean(ref_del)), 'ref_deletion_auc_std': float(np.std(ref_del)),
            'ref_insertion_auc_mean': float(np.mean(ref_ins)), 'ref_insertion_auc_std': float(np.std(ref_ins)),
        },
    }

    if include_shap:
        print("Computing SHAP attributions (DP + reference, this is the slow step)...")
        rng = np.random.RandomState(42)
        bg_idx = rng.choice(len(images_all), size=min(20, len(images_all)), replace=False)
        background = images_all[bg_idx]
        shap_n = min(50, len(images))
        shap_slice_idx = rng.choice(len(images), size=shap_n, replace=False)

        dp_shap = compute_shap_batch(dp_model, images[shap_slice_idx], background, device)
        ref_shap = compute_shap_batch(ref_model, images[shap_slice_idx], background, device)

        shap_sims = []
        for i, slice_i in enumerate(shap_slice_idx):
            cls = int(labels[slice_i])
            dp_map = dp_shap[i, :, :, cls]
            ref_map = ref_shap[i, :, :, cls]
            dp_norm = (dp_map - dp_map.min()) / (dp_map.max() - dp_map.min() + 1e-8)
            ref_norm = (ref_map - ref_map.min()) / (ref_map.max() - ref_map.min() + 1e-8)
            shap_sims.append(heatmap_similarity(dp_norm, ref_norm))

        results['shap_similarity'] = {
            'ssim_mean': float(np.mean([s['ssim'] for s in shap_sims])),
            'spearman_mean': float(np.nanmean([s['spearman'] for s in shap_sims])),
            'dice_top20_mean': float(np.mean([s['dice_top20'] for s in shap_sims])),
            'n_slices': shap_n,
        }

    os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
    dp_tag = os.path.splitext(os.path.basename(dp_checkpoint))[0]
    out_path = os.path.join(results_dir, 'metrics', f'{dp_tag}_xai_similarity.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ XAI similarity results saved: {out_path}")
    print(f"  Grad-CAM SSIM={results['gradcam_similarity']['ssim_mean']:.3f}  "
          f"Spearman={results['gradcam_similarity']['spearman_mean']:.3f}  "
          f"Dice@20%={results['gradcam_similarity']['dice_top20_mean']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='PPXFL explanation-degradation-under-DP analysis')
    parser.add_argument('--dp-checkpoint', type=str, required=True)
    parser.add_argument('--reference-checkpoint', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, choices=['vgg19', 'resnet50'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--splits-path', type=str, default=None)
    parser.add_argument('--freeze-backbone-dp', action='store_true')
    parser.add_argument('--freeze-backbone-ref', action='store_true')
    parser.add_argument('--dp-scope', type=str, default='none', choices=['none', 'head', 'full'],
                        help='"full" loads the GroupNorm-converted architecture for the DP checkpoint')
    parser.add_argument('--max-slices', type=int, default=None)
    parser.add_argument('--include-shap', action='store_true')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'processed')
    if args.splits_path is None:
        args.splits_path = os.path.join(project_root, 'data', 'splits', 'splits_v1.json')
    results_dir = os.path.join(project_root, 'results')

    run_xai_similarity(
        args.dp_checkpoint, args.reference_checkpoint, args.model,
        args.data_dir, args.splits_path, results_dir, fold=args.fold,
        freeze_backbone_dp=args.freeze_backbone_dp, freeze_backbone_ref=args.freeze_backbone_ref,
        dp_scope=args.dp_scope, max_slices=args.max_slices, include_shap=args.include_shap,
    )


if __name__ == '__main__':
    main()
