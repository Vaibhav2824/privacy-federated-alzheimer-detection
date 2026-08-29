"""
mia.py — Membership Inference Attack suite for PPXFL.

Replaces the confidence-threshold attack that used to live in evaluate.py. That
attack tuned its threshold on the very data being attacked and re-derived
"membership" with a fresh np.random.permutation that did not match the actual
training split — which is why it reported the same ~88.9%/38.9% numbers for
two different architectures. Both bugs are fixed here:

  - Membership ground truth is read ONLY from a run's `*_run_meta.json`
    (written by centralised_train.py / fl_server.py / dp_train.py at train
    time), never re-derived.
  - Attack thresholds are calibrated on shadow data, not on the target's own
    evaluation split.

Two attacks:
  1. Yeom loss-threshold attack (Yeom et al. 2018) — zero extra training cost.
     Score = -mean per-slice CE loss, aggregated per subject. Lower loss under
     the target model suggests membership.
  2. Shadow-trained attack classifier (Shokri et al. 2017 style). We deviate
     from literal per-example LiRA (Carlini et al. 2022): LiRA needs many
     shadow models to have separately seen EACH example in and out to fit a
     per-example Gaussian, which is not viable with n~32-subject cohorts and
     a ~16-shadow budget. Instead we pool (loss, confidence, entropy)
     features from shadow in/out subjects into one logistic-regression attack
     classifier — a documented, honest simplification of the plan's
     "LiRA-lite" label.

Both attacks operate at SUBJECT granularity: per-slice scores are averaged to
one score per subject before computing ROC/AUC, because attacking at the
slice level overstates success (a subject with N scans just gives the
attacker N correlated guesses).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centralised_train import MRIDataset
from models import get_model
from splits import load_split


def _per_slice_scores(model, images, labels, device, batch_size=32):
    """Return per-slice (loss, confidence, entropy) under `model`, no grad."""
    model.eval()
    ds = MRIDataset(images, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    ce = nn.CrossEntropyLoss(reduction='none')

    losses, confs, ents = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            loss = ce(logits, lbls)
            probs = torch.softmax(logits, dim=1)
            conf, _ = probs.max(dim=1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

            losses.extend(loss.cpu().numpy())
            confs.extend(conf.cpu().numpy())
            ents.extend(entropy.cpu().numpy())

    return np.array(losses), np.array(confs), np.array(ents)


def _aggregate_per_subject(scores, subject_ids):
    """Mean + std of a per-slice score array, grouped by subject_id (aligned arrays)."""
    subject_ids = np.asarray(subject_ids)
    unique_subjects = sorted(set(subject_ids.tolist()))
    means, stds = [], []
    for s in unique_subjects:
        vals = scores[subject_ids == s]
        means.append(vals.mean())
        stds.append(vals.std() if len(vals) > 1 else 0.0)
    return unique_subjects, np.array(means), np.array(stds)


def _subject_ids_for_indices(manifest, array_indices):
    idx_to_subject = manifest.set_index('array_index')['subject_id']
    return idx_to_subject.loc[array_indices].to_numpy()


def yeom_attack(model, images, labels, manifest, member_idx, nonmember_idx, device):
    """Loss-threshold attack, subject-aggregated. Returns dict with AUC/TPR@lowFPR."""
    member_loss, _, _ = _per_slice_scores(model, images[member_idx], labels[member_idx], device)
    nonmember_loss, _, _ = _per_slice_scores(model, images[nonmember_idx], labels[nonmember_idx], device)

    member_subj = _subject_ids_for_indices(manifest, member_idx)
    nonmember_subj = _subject_ids_for_indices(manifest, nonmember_idx)

    m_subjects, m_mean_loss, _ = _aggregate_per_subject(member_loss, member_subj)
    n_subjects, n_mean_loss, _ = _aggregate_per_subject(nonmember_loss, nonmember_subj)

    scores = np.concatenate([-m_mean_loss, -n_mean_loss])  # higher score = more likely member
    y_true = np.concatenate([np.ones(len(m_subjects)), np.zeros(len(n_subjects))])

    return _summarise_attack(y_true, scores, n_members=len(m_subjects), n_nonmembers=len(n_subjects))


def _summarise_attack(y_true, scores, n_members, n_nonmembers):
    if len(set(y_true.tolist())) < 2:
        return {'error': 'degenerate: only one class present', 'n_members': n_members, 'n_nonmembers': n_nonmembers}

    auc = roc_auc_score(y_true, scores)
    fpr, tpr, thresh = roc_curve(y_true, scores)

    def tpr_at_fpr(target_fpr):
        idx = np.searchsorted(fpr, target_fpr, side='right') - 1
        idx = max(idx, 0)
        return float(tpr[idx])

    balanced_acc = max((tpr[i] + (1 - fpr[i])) / 2 for i in range(len(fpr)))

    return {
        'auc': float(auc),
        'tpr_at_fpr_0.1': tpr_at_fpr(0.1),
        'tpr_at_fpr_0.01': tpr_at_fpr(0.01),
        'balanced_accuracy': float(balanced_acc),
        'n_members': int(n_members),
        'n_nonmembers': int(n_nonmembers),
        'roc_fpr': fpr.tolist(),
        'roc_tpr': tpr.tolist(),
    }


def train_shadow_models(model_name, dp_scope_arch, attacker_subjects, manifest, images, labels,
                        device, num_shadows=16, epochs=8, seed=42):
    """Train `num_shadows` shadow models on random 50/50 subject splits of the
    attacker's known population, collecting (loss, conf, entropy) features
    labeled by shadow-membership for the attack classifier.

    `dp_scope_arch` controls whether shadows use the head-only architecture
    (freeze_backbone=True) — cheap, and matches head-scope DP targets. Full-
    scope targets reuse the same shadows as an approximation (see mia.py
    module docstring / plan risk notes).
    """
    from centralised_train import compute_class_weights, train_one_epoch

    rng = np.random.RandomState(seed)
    attacker_subjects = sorted(attacker_subjects)

    feature_rows = []  # (mean_loss, std_loss, mean_conf, mean_entropy, label)

    for shadow_i in range(num_shadows):
        shuffled = attacker_subjects.copy()
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        in_subjects = set(shuffled[:half])
        out_subjects = set(shuffled[half:])

        in_idx = manifest.index[manifest['subject_id'].isin(in_subjects)]
        in_idx = manifest.loc[in_idx, 'array_index'].to_numpy()
        out_idx = manifest.index[manifest['subject_id'].isin(out_subjects)]
        out_idx = manifest.loc[out_idx, 'array_index'].to_numpy()

        shadow_model = get_model(model_name, num_classes=3, pretrained=True,
                                 freeze_backbone=dp_scope_arch).to(device)
        train_ds = MRIDataset(images[in_idx], labels[in_idx], augment=True)
        loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
        class_weights = compute_class_weights(labels[in_idx]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, shadow_model.parameters()), lr=1e-4)

        for _ in range(epochs):
            train_one_epoch(shadow_model, loader, criterion, optimizer, device)

        in_loss, in_conf, in_ent = _per_slice_scores(shadow_model, images[in_idx], labels[in_idx], device)
        out_loss, out_conf, out_ent = _per_slice_scores(shadow_model, images[out_idx], labels[out_idx], device)

        in_subj_ids = _subject_ids_for_indices(manifest, in_idx)
        out_subj_ids = _subject_ids_for_indices(manifest, out_idx)

        for scores, subj_ids, label in [((in_loss, in_conf, in_ent), in_subj_ids, 1),
                                         ((out_loss, out_conf, out_ent), out_subj_ids, 0)]:
            loss_arr, conf_arr, ent_arr = scores
            subjects, mean_loss, std_loss = _aggregate_per_subject(loss_arr, subj_ids)
            _, mean_conf, _ = _aggregate_per_subject(conf_arr, subj_ids)
            _, mean_ent, _ = _aggregate_per_subject(ent_arr, subj_ids)
            for i in range(len(subjects)):
                feature_rows.append([mean_loss[i], std_loss[i], mean_conf[i], mean_ent[i], label])

        del shadow_model
        torch.cuda.empty_cache()
        print(f"  shadow {shadow_i+1}/{num_shadows}: {len(in_subjects)} in / {len(out_subjects)} out subjects")

    feature_rows = np.array(feature_rows)
    X, y = feature_rows[:, :4], feature_rows[:, 4]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    train_auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    print(f"  shadow attack classifier fit on {len(y)} subject-rows, in-sample AUC={train_auc:.3f}")
    return clf


def shadow_attack(clf, model, images, labels, manifest, member_idx, nonmember_idx, device):
    """Apply a pre-fit shadow attack classifier to a target model's member/nonmember subjects."""
    member_loss, member_conf, member_ent = _per_slice_scores(model, images[member_idx], labels[member_idx], device)
    nonmember_loss, nonmember_conf, nonmember_ent = _per_slice_scores(model, images[nonmember_idx], labels[nonmember_idx], device)

    member_subj = _subject_ids_for_indices(manifest, member_idx)
    nonmember_subj = _subject_ids_for_indices(manifest, nonmember_idx)

    def featurize(loss_arr, conf_arr, ent_arr, subj_ids):
        subjects, mean_loss, std_loss = _aggregate_per_subject(loss_arr, subj_ids)
        _, mean_conf, _ = _aggregate_per_subject(conf_arr, subj_ids)
        _, mean_ent, _ = _aggregate_per_subject(ent_arr, subj_ids)
        X = np.stack([mean_loss, std_loss, mean_conf, mean_ent], axis=1)
        return subjects, X

    m_subjects, Xm = featurize(member_loss, member_conf, member_ent, member_subj)
    n_subjects, Xn = featurize(nonmember_loss, nonmember_conf, nonmember_ent, nonmember_subj)

    scores = np.concatenate([clf.predict_proba(Xm)[:, 1], clf.predict_proba(Xn)[:, 1]])
    y_true = np.concatenate([np.ones(len(m_subjects)), np.zeros(len(n_subjects))])

    return _summarise_attack(y_true, scores, n_members=len(m_subjects), n_nonmembers=len(n_subjects))


def _resolve_train_subjects(run_meta):
    """centralised_train.py/dp_train.py's run_meta stores 'train_subjects' directly.
    fl_server.py's run_meta has no single train set (each client trains on its own
    slice) — instead it carries 'client_data_dir' pointing at partition.py's output.
    Membership for an FL target is "trained by ANY client", so union client_subjects."""
    if 'train_subjects' in run_meta:
        return run_meta['train_subjects']
    if 'client_data_dir' in run_meta:
        meta_path = os.path.join(run_meta['client_data_dir'], 'partition_metadata.json')
        with open(meta_path) as f:
            partition_meta = json.load(f)
        all_subjects = []
        for subjects in partition_meta['client_subjects'].values():
            all_subjects.extend(subjects)
        return all_subjects
    raise KeyError("run_meta has neither 'train_subjects' nor 'client_data_dir'")


def run_mia(checkpoint_path, run_meta_path, model_name, manifest_path, data_dir,
           results_dir, shadow_clf=None, freeze_backbone_for_target=False, dp_scope='none', device=None):
    """Run both attacks against one target checkpoint. Membership truth from run_meta.json only."""
    import pandas as pd

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = pd.read_csv(manifest_path)
    images = np.load(os.path.join(data_dir, 'all_images.npy'))
    labels = np.load(os.path.join(data_dir, 'all_labels.npy'))

    with open(run_meta_path) as f:
        run_meta = json.load(f)
    run_tag = run_meta['run_tag']

    train_subjects = set(_resolve_train_subjects(run_meta))
    test_subjects = set(run_meta['test_subjects'])
    train_idx = manifest.loc[manifest['subject_id'].isin(train_subjects), 'array_index'].to_numpy()
    test_idx = manifest.loc[manifest['subject_id'].isin(test_subjects), 'array_index'].to_numpy()

    if dp_scope == 'full':
        # Full-scope DP checkpoints have BatchNorm converted to GroupNorm
        # (Opacus ModuleValidator.fix, see dp_train.py) — a plain get_model()
        # BatchNorm architecture can't load their state_dict at all.
        from dp_train import build_dp_model_and_optimizer
        model, _ = build_dp_model_and_optimizer(model_name, 'full', lr=1e-4, weight_decay=1e-4)
    else:
        model = get_model(model_name, num_classes=3, pretrained=False, freeze_backbone=freeze_backbone_for_target)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    del ckpt, state_dict  # loaded with map_location=device — free the duplicate before the attack loop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nMIA on {run_tag}: {len(train_subjects)} member subjects / {len(test_subjects)} non-member subjects")

    results = {'run_tag': run_tag, 'checkpoint': checkpoint_path}
    results['yeom'] = yeom_attack(model, images, labels, manifest, train_idx, test_idx, device)
    if shadow_clf is not None:
        results['shadow'] = shadow_attack(shadow_clf, model, images, labels, manifest, train_idx, test_idx, device)

    os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
    out_path = os.path.join(results_dir, 'metrics', f'{run_tag}_mia_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ MIA results saved: {out_path}")
    print(f"  Yeom: AUC={results['yeom'].get('auc', float('nan')):.3f}  "
          f"TPR@FPR0.01={results['yeom'].get('tpr_at_fpr_0.01', float('nan')):.3f}")
    if shadow_clf is not None:
        print(f"  Shadow: AUC={results['shadow'].get('auc', float('nan')):.3f}  "
              f"TPR@FPR0.01={results['shadow'].get('tpr_at_fpr_0.01', float('nan')):.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='PPXFL Membership Inference Attack suite')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--run-meta', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, choices=['vgg19', 'resnet50'])
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--splits-path', type=str, default=None)
    parser.add_argument('--fold', type=int, default=0, help='Fold whose train+val subjects form the shadow-model population')
    parser.add_argument('--num-shadows', type=int, default=16)
    parser.add_argument('--shadow-epochs', type=int, default=8)
    parser.add_argument('--freeze-target-backbone', action='store_true',
                        help='Set if the target checkpoint was trained with a frozen backbone (head-only DP)')
    parser.add_argument('--dp-scope', type=str, default='none', choices=['none', 'head', 'full'],
                        help='"full" loads the GroupNorm-converted architecture full-scope DP '
                             'checkpoints actually use (plain BatchNorm can\'t load their state_dict)')
    parser.add_argument('--skip-shadow', action='store_true', help='Only run the (cheap) Yeom attack')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-shadow', type=str, default=None,
                        help='Pickle the fitted shadow attack classifier here after training')
    parser.add_argument('--load-shadow', type=str, default=None,
                        help='Load a previously pickled shadow classifier instead of retraining '
                             '16 shadow models from scratch (the 16-shadow budget is meant to be '
                             'a one-time cost per plan, reused across every target checkpoint)')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'processed')
    if args.splits_path is None:
        args.splits_path = os.path.join(project_root, 'data', 'splits', 'splits_v1.json')
    results_dir = os.path.join(project_root, 'results')
    manifest_path = os.path.join(args.data_dir, 'manifest.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    shadow_clf = None
    if args.load_shadow:
        import pickle
        with open(args.load_shadow, 'rb') as f:
            shadow_clf = pickle.load(f)
        print(f"Loaded cached shadow attack classifier from {args.load_shadow}")
    elif not args.skip_shadow:
        import pandas as pd
        manifest = pd.read_csv(manifest_path)
        images = np.load(os.path.join(args.data_dir, 'all_images.npy'))
        labels = np.load(os.path.join(args.data_dir, 'all_labels.npy'))
        split = load_split(args.fold, manifest_path, args.splits_path)
        attacker_subjects = split['train_subjects'] + split['val_subjects']

        print(f"Training {args.num_shadows} shadow models on {len(attacker_subjects)} "
              f"attacker-known subjects (fold {args.fold} train+val)...")
        shadow_clf = train_shadow_models(
            args.model, args.freeze_target_backbone, attacker_subjects, manifest, images, labels,
            device, num_shadows=args.num_shadows, epochs=args.shadow_epochs, seed=args.seed,
        )
        if args.save_shadow:
            import pickle
            os.makedirs(os.path.dirname(args.save_shadow), exist_ok=True)
            with open(args.save_shadow, 'wb') as f:
                pickle.dump(shadow_clf, f)
            print(f"  Saved shadow attack classifier to {args.save_shadow}")

    run_mia(args.checkpoint, args.run_meta, args.model, manifest_path, args.data_dir,
           results_dir, shadow_clf=shadow_clf, freeze_backbone_for_target=args.freeze_target_backbone,
           dp_scope=args.dp_scope, device=device)


if __name__ == '__main__':
    main()
