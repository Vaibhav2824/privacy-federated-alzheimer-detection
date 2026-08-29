"""
splits.py — Subject-level cross-validation splits for PPXFL.

The raw ADNI cohort has far fewer unique subjects than scans (each subject was
imaged repeatedly). Splitting at the scan or slice level lets the same
subject's brain appear in both train and test, which leaks identity-specific
signal and inflates every downstream metric. This module builds the split
once, at the subject level, and every training/evaluation script must load it
from here rather than re-splitting on its own.

Usage:
    python -m src.splits --manifest data/processed/manifest.csv \
        --out data/splits/splits_v1.json --k 5 --seed 42

    from src.splits import load_split
    split = load_split(fold=0, manifest_path='data/processed/manifest.csv',
                        splits_path='data/splits/splits_v1.json')
    split['train_idx'], split['val_idx'], split['test_idx']  # array indices
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _subject_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per subject: subject_id, label (must be constant within subject)."""
    grouped = manifest.groupby('subject_id')['label']
    nunique = grouped.nunique()
    bad = nunique[nunique != 1]
    if len(bad) > 0:
        raise ValueError(f"Subjects with inconsistent class labels across scans: {bad.index.tolist()}")
    first_label = grouped.first()
    subjects = pd.DataFrame({
        'subject_id': first_label.index,
        'label': first_label.values,
    }).reset_index(drop=True)
    return subjects


def build_splits(manifest_path: str, out_path: str, k: int = 5, val_subjects_per_fold: int = 2,
                  seed: int = 42) -> dict:
    manifest = pd.read_csv(manifest_path)
    subjects = _subject_table(manifest)

    if len(subjects) < k:
        raise ValueError(f"Only {len(subjects)} unique subjects but k={k} folds requested — reduce k.")

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    X = np.zeros(len(subjects))  # unused, StratifiedGroupKFold only needs shapes
    y = subjects['label'].values
    groups = subjects['subject_id'].values

    folds = {}
    rng = np.random.RandomState(seed)

    for fold_i, (trainval_pos, test_pos) in enumerate(sgkf.split(X, y, groups)):
        trainval_subjects = subjects.iloc[trainval_pos].reset_index(drop=True)
        test_subjects = subjects.iloc[test_pos]['subject_id'].tolist()

        # Carve `val_subjects_per_fold` per class off trainval, stratified.
        val_ids = []
        for label in sorted(trainval_subjects['label'].unique()):
            class_subj = trainval_subjects[trainval_subjects['label'] == label]['subject_id'].tolist()
            n_val = min(val_subjects_per_fold, max(1, len(class_subj) // 5))
            rng.shuffle(class_subj)
            val_ids.extend(class_subj[:n_val])

        train_ids = [s for s in trainval_subjects['subject_id'].tolist() if s not in set(val_ids)]

        overlap_tv = set(train_ids) & set(val_ids)
        overlap_tt = (set(train_ids) | set(val_ids)) & set(test_subjects)
        assert not overlap_tv, f"fold {fold_i}: train/val subject overlap {overlap_tv}"
        assert not overlap_tt, f"fold {fold_i}: train+val/test subject overlap {overlap_tt}"

        folds[str(fold_i)] = {
            'train_subjects': sorted(train_ids),
            'val_subjects': sorted(val_ids),
            'test_subjects': sorted(test_subjects),
        }

    result = {
        'k': k,
        'seed': seed,
        'n_subjects': len(subjects),
        'class_counts': subjects['label'].value_counts().to_dict(),
        'folds': folds,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else o)

    return result


def load_split(fold: int, manifest_path: str, splits_path: str) -> dict:
    """Return array indices (into all_images.npy / all_labels.npy) for one fold."""
    manifest = pd.read_csv(manifest_path)
    with open(splits_path) as f:
        splits = json.load(f)

    fold_def = splits['folds'][str(fold)]
    result = {'fold': fold}
    for part in ('train', 'val', 'test'):
        subject_set = set(fold_def[f'{part}_subjects'])
        idx = manifest.index[manifest['subject_id'].isin(subject_set)].to_numpy()
        result[f'{part}_idx'] = manifest.loc[idx, 'array_index'].to_numpy()
        result[f'{part}_subjects'] = fold_def[f'{part}_subjects']

    all_idx = np.concatenate([result['train_idx'], result['val_idx'], result['test_idx']])
    assert len(all_idx) == len(set(all_idx.tolist())), f"fold {fold}: duplicate array indices across splits"

    return result


def verify_splits(manifest_path: str, splits_path: str, k: int = 5) -> None:
    """Sanity check: no subject crosses partitions in any fold, every fold covers all subjects."""
    manifest = pd.read_csv(manifest_path)
    all_subjects = set(manifest['subject_id'].unique())

    for fold in range(k):
        s = load_split(fold, manifest_path, splits_path)
        train_s, val_s, test_s = set(s['train_subjects']), set(s['val_subjects']), set(s['test_subjects'])
        assert not (train_s & val_s), f"fold {fold}: train/val overlap"
        assert not (train_s & test_s), f"fold {fold}: train/test overlap"
        assert not (val_s & test_s), f"fold {fold}: val/test overlap"
        covered = train_s | val_s | test_s
        assert covered == all_subjects, f"fold {fold}: subjects missing from split: {all_subjects - covered}"
        print(f"  fold {fold}: train={len(train_s)} subj / val={len(val_s)} subj / test={len(test_s)} subj "
              f"({len(s['train_idx'])}/{len(s['val_idx'])}/{len(s['test_idx'])} slices) — OK")

    print(f"verify_splits: all {k} folds OK, {len(all_subjects)} subjects total")


def main():
    parser = argparse.ArgumentParser(description='Build subject-level GroupKFold splits')
    parser.add_argument('--manifest', type=str, default='data/processed/manifest.csv')
    parser.add_argument('--out', type=str, default='data/splits/splits_v1.json')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify', action='store_true', help='Verify an existing splits file instead of building')
    args = parser.parse_args()

    if args.verify:
        verify_splits(args.manifest, args.out, args.k)
        return

    result = build_splits(args.manifest, args.out, k=args.k, seed=args.seed)
    print(f"Built {result['k']}-fold subject-level splits from {result['n_subjects']} subjects")
    print(f"Class counts: {result['class_counts']}")
    for fold_i, fold in result['folds'].items():
        print(f"  fold {fold_i}: train={len(fold['train_subjects'])} val={len(fold['val_subjects'])} "
              f"test={len(fold['test_subjects'])} subjects")

    verify_splits(args.manifest, args.out, k=args.k)


if __name__ == '__main__':
    main()
