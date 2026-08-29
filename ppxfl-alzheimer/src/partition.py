"""
partition.py — Dirichlet Non-IID Data Partitioning
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Splits preprocessed dataset across K simulated hospital clients using
Dirichlet(α) distribution to create realistic non-IID splits.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import load_split


def dirichlet_partition_subjects(subject_ids, subject_labels, num_clients=4, alpha=0.5, seed=42):
    """
    Partition SUBJECTS (not slices/scans) using a Dirichlet(alpha) distribution.

    Every slice belonging to a subject must land on the same client — partitioning
    at the slice level would let one subject's scans leak across multiple
    "hospitals", which is both unrealistic (a patient's MRI would be at one site)
    and a data-leakage risk once FL evaluation crosses clients.

    Args:
        subject_ids: array-like of subject id strings
        subject_labels: array-like of int labels, same order as subject_ids
        num_clients: K — number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed for reproducibility

    Returns:
        client_subjects: dict mapping client_id -> list of subject_id strings

    At small subject counts (this dataset has ~22-23 train subjects per fold
    for K=4-8 clients), a single Dirichlet draw can leave a client with ZERO
    subjects — which would later crash training (division by zero over an
    empty DataLoader). We detect that and retry with a derived seed, up to
    `max_attempts`, before giving up loudly instead of producing a partition
    that silently breaks a later script.
    """
    subject_ids = np.asarray(subject_ids)
    subject_labels = np.asarray(subject_labels)
    num_classes = len(np.unique(subject_labels))

    if len(subject_ids) < num_clients:
        raise ValueError(f"Cannot partition {len(subject_ids)} subjects across {num_clients} "
                         f"clients — fewer subjects than clients.")

    max_attempts = 50
    client_indices = None
    for attempt in range(max_attempts):
        rng = np.random.RandomState(seed + attempt * 7919)  # 7919 is prime, avoids seed collisions
        class_indices = {c: np.where(subject_labels == c)[0] for c in range(num_classes)}
        trial_indices = {k: [] for k in range(num_clients)}

        for c in range(num_classes):
            indices = class_indices[c].copy()
            rng.shuffle(indices)

            proportions = rng.dirichlet([alpha] * num_clients)
            proportions = np.maximum(proportions, 0.01)
            proportions = proportions / proportions.sum()

            split_points = (np.cumsum(proportions) * len(indices)).astype(int)
            split_points[-1] = len(indices)

            prev = 0
            for k in range(num_clients):
                trial_indices[k].extend(indices[prev:split_points[k]].tolist())
                prev = split_points[k]

        if all(len(v) > 0 for v in trial_indices.values()):
            client_indices = trial_indices
            break

    if client_indices is None:
        raise RuntimeError(
            f"Could not find a Dirichlet(alpha={alpha}) partition with every client "
            f"non-empty after {max_attempts} attempts ({len(subject_ids)} subjects, "
            f"K={num_clients}). Try a larger alpha or fewer clients."
        )

    client_subjects = {}
    for k in range(num_clients):
        idx = np.array(client_indices[k], dtype=int)
        rng.shuffle(idx)
        client_subjects[k] = subject_ids[idx].tolist()

    return client_subjects


def expand_subjects_to_indices(client_subjects, manifest):
    """Map each client's subject list to array indices (into all_images.npy) via the manifest."""
    client_indices = {}
    for k, subj_list in client_subjects.items():
        subj_set = set(subj_list)
        idx = manifest.index[manifest['subject_id'].isin(subj_set)]
        client_indices[k] = manifest.loc[idx, 'array_index'].to_numpy()
    return client_indices


def create_client_datasets(images, labels, client_indices, output_dir):
    """
    Save partitioned data to client directories.

    Args:
        images: numpy array of all images
        labels: numpy array of all labels
        client_indices: dict from dirichlet_partition()
        output_dir: path to clients/ directory
    """
    class_names = {0: 'CN', 1: 'MCI', 2: 'AD'}

    for client_id, indices in client_indices.items():
        client_dir = os.path.join(output_dir, f'client_{client_id + 1}')

        # Create class subdirectories
        for class_name in class_names.values():
            os.makedirs(os.path.join(client_dir, class_name), exist_ok=True)

        client_images = images[indices]
        client_labels = labels[indices]

        # Save images per class
        class_counters = {0: 0, 1: 0, 2: 0}
        for img, lbl in zip(client_images, client_labels):
            class_name = class_names[int(lbl)]
            fname = f'{class_name}_{class_counters[int(lbl)]:04d}.npy'
            np.save(os.path.join(client_dir, class_name, fname), img)
            class_counters[int(lbl)] += 1

        # Save combined arrays for easy loading
        np.save(os.path.join(client_dir, 'images.npy'), client_images)
        np.save(os.path.join(client_dir, 'labels.npy'), client_labels)


def compute_partition_stats(labels, client_indices, class_names=None):
    """Compute and print partition statistics."""
    if class_names is None:
        class_names = {0: 'CN', 1: 'MCI', 2: 'AD'}

    num_clients = len(client_indices)
    num_classes = len(class_names)

    stats = {}

    print(f"\n{'='*70}")
    print(f"Partition Statistics (K={num_clients} clients)")
    print(f"{'='*70}")

    header = f"{'Client':<12}"
    for c in range(num_classes):
        header += f"{class_names[c]:<10}"
    header += f"{'Total':<10}{'% of Data':<12}"
    print(header)
    print('-' * 70)

    # Denominator is the pool actually being partitioned (this fold's train
    # subjects), NOT the full dataset — labels[] is the global array and most
    # of it (val/test slices, other folds' train slices) was never a candidate.
    total_samples = sum(len(v) for v in client_indices.values())

    for k in range(num_clients):
        indices = client_indices[k]
        client_labels = labels[indices]

        client_stats = {}
        row = f"Client {k+1:<5}"

        for c in range(num_classes):
            count = int(np.sum(client_labels == c))
            client_stats[class_names[c]] = count
            row += f"{count:<10}"

        total = len(indices)
        pct = 100.0 * total / total_samples
        client_stats['total'] = total
        client_stats['percentage'] = round(pct, 1)
        row += f"{total:<10}{pct:.1f}%"
        print(row)

        stats[f'client_{k+1}'] = client_stats

    print('-' * 70)

    # Overall row — summed across clients, not the global label array
    row = f"{'Total':<12}"
    for c in range(num_classes):
        count = sum(int(np.sum(labels[client_indices[k]] == c)) for k in range(num_clients))
        row += f"{count:<10}"
    row += f"{total_samples:<10}{'100.0%':<12}"
    print(row)

    return stats


def plot_partition_distribution(labels, client_indices, save_path=None):
    """Create a stacked bar chart of class distribution per client."""
    class_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
    colors = {'CN': '#2196F3', 'MCI': '#FF9800', 'AD': '#F44336'}

    num_clients = len(client_indices)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Non-IID Data Partition Distribution', fontsize=14, fontweight='bold')

    # Stacked bar chart
    x = np.arange(num_clients)
    width = 0.6
    bottom = np.zeros(num_clients)

    for c, name in class_names.items():
        counts = [np.sum(labels[client_indices[k]] == c) for k in range(num_clients)]
        ax1.bar(x, counts, width, label=name, bottom=bottom, color=colors[name], edgecolor='white')
        bottom += counts

    ax1.set_xlabel('Client', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Samples per Client (Absolute)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Client {k+1}' for k in range(num_clients)])
    ax1.legend()

    # Percentage stacked bar chart
    bottom_pct = np.zeros(num_clients)
    for c, name in class_names.items():
        counts = np.array([np.sum(labels[client_indices[k]] == c) for k in range(num_clients)])
        totals = np.array([len(client_indices[k]) for k in range(num_clients)])
        pcts = 100.0 * counts / (totals + 1e-8)
        ax2.bar(x, pcts, width, label=name, bottom=bottom_pct, color=colors[name], edgecolor='white')
        bottom_pct += pcts

    ax2.set_xlabel('Client', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Class Distribution per Client (%)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Client {k+1}' for k in range(num_clients)])
    ax2.set_ylim(0, 105)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Partition plot saved to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PPXFL Data Partitioning (subject-level, fold-scoped)')
    parser.add_argument('--processed-dir', type=str, default=None,
                        help='Directory with preprocessed data (all_images.npy, all_labels.npy, manifest.csv)')
    parser.add_argument('--splits', type=str, default=None,
                        help='Path to splits_v1.json (default: data/splits/splits_v1.json)')
    parser.add_argument('--fold', type=int, required=True,
                        help='Fold index — clients are drawn ONLY from this fold\'s train subjects')
    parser.add_argument('--output-root', type=str, default=None,
                        help='Root output directory for client partitions (default: data/clients)')
    parser.add_argument('--num-clients', type=int, default=4,
                        help='Number of federated clients K (default: 4)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha (default: 0.5, lower = more non-IID)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='Where to write the partition figure (default: results/figures). '
                             'Set this when partitioning for a results directory other than '
                             'the default one, so the figure lands beside its own run.')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.processed_dir is None:
        args.processed_dir = os.path.join(project_root, 'data', 'processed')
    if args.splits is None:
        args.splits = os.path.join(project_root, 'data', 'splits', 'splits_v1.json')
    if args.output_root is None:
        args.output_root = os.path.join(project_root, 'data', 'clients')
    args.output = os.path.join(args.output_root, f'f{args.fold}_a{args.alpha}_s{args.seed}')

    images_path = os.path.join(args.processed_dir, 'all_images.npy')
    labels_path = os.path.join(args.processed_dir, 'all_labels.npy')
    manifest_path = os.path.join(args.processed_dir, 'manifest.csv')

    for p in (images_path, labels_path, manifest_path, args.splits):
        if not os.path.exists(p):
            print(f"[ERROR] {p} not found. Run preprocess.py and splits.py first.")
            sys.exit(1)

    print("Loading preprocessed data + manifest + splits...")
    images = np.load(images_path)
    labels = np.load(labels_path)
    manifest = pd.read_csv(manifest_path)
    split = load_split(args.fold, manifest_path, args.splits)
    print(f"  Fold {args.fold}: {len(split['train_subjects'])} train subjects "
          f"({len(split['train_idx'])} slices) available for FL partitioning "
          f"(val/test subjects excluded: {len(split['val_subjects'])}/{len(split['test_subjects'])})")

    # Subject-level table restricted to this fold's train subjects
    train_manifest = manifest[manifest['subject_id'].isin(split['train_subjects'])]
    subj_table = train_manifest.drop_duplicates('subject_id')[['subject_id', 'label']]

    print(f"\nPartitioning {len(subj_table)} train subjects with Dirichlet(α={args.alpha}), "
          f"K={args.num_clients} clients...")
    client_subjects = dirichlet_partition_subjects(
        subj_table['subject_id'].values,
        subj_table['label'].values,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
    )
    client_indices = expand_subjects_to_indices(client_subjects, manifest)

    # Guard: no client subject set may intersect val/test subjects of this fold
    holdout = set(split['val_subjects']) | set(split['test_subjects'])
    for k, subj_list in client_subjects.items():
        leaked = set(subj_list) & holdout
        assert not leaked, f"client {k} contains held-out subjects: {leaked}"

    stats = compute_partition_stats(labels, client_indices)

    print(f"\nSaving client datasets to {args.output}...")
    create_client_datasets(images, labels, client_indices, args.output)

    metadata = {
        'fold': args.fold,
        'num_clients': args.num_clients,
        'alpha': args.alpha,
        'seed': args.seed,
        'train_subjects_available': len(subj_table),
        'total_samples_partitioned': int(sum(len(v) for v in client_indices.values())),
        'client_subjects': {str(k): v for k, v in client_subjects.items()},
        'clients': stats,
    }

    meta_path = os.path.join(args.output, 'partition_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved to {meta_path}")

    fig_dir = args.figures_dir or os.path.join(project_root, 'results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plot_partition_distribution(
        labels, client_indices,
        save_path=os.path.join(fig_dir, f'partition_f{args.fold}_alpha{args.alpha}.png')
    )

    partition_path = os.path.join(args.output, 'partition_indices.json')
    partition_data = {str(k): client_indices[k].tolist() for k in client_indices}
    with open(partition_path, 'w') as f:
        json.dump(partition_data, f)
    print(f"  ✓ Partition indices saved to {partition_path}")

    print(f"\n{'='*60}")
    print("Partitioning complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
