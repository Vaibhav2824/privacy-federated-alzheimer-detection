"""
partition.py — Dirichlet Non-IID Data Partitioning
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Splits preprocessed dataset across K simulated hospital clients using 
Dirichlet(α) distribution to create realistic non-IID splits.
"""

import os
import sys
import json
import argparse
import numpy as np
import shutil
from pathlib import Path
import matplotlib.pyplot as plt


def dirichlet_partition(labels, num_clients=4, alpha=0.5, seed=42):
    """
    Partition dataset indices using Dirichlet distribution.
    
    Args:
        labels: numpy array of integer labels
        num_clients: K — number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed for reproducibility
    
    Returns:
        client_indices: dict mapping client_id -> array of sample indices
    """
    np.random.seed(seed)
    num_classes = len(np.unique(labels))
    
    # Get indices for each class
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
    
    client_indices = {k: [] for k in range(num_clients)}
    
    for c in range(num_classes):
        indices = class_indices[c].copy()
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Ensure minimum samples per client (at least 1)
        proportions = np.maximum(proportions, 0.01)
        proportions = proportions / proportions.sum()
        
        # Split indices according to proportions
        split_points = (np.cumsum(proportions) * len(indices)).astype(int)
        split_points[-1] = len(indices)  # Ensure all indices used
        
        prev = 0
        for k in range(num_clients):
            client_indices[k].extend(indices[prev:split_points[k]].tolist())
            prev = split_points[k]
    
    # Convert to numpy arrays
    for k in range(num_clients):
        client_indices[k] = np.array(client_indices[k])
        np.random.shuffle(client_indices[k])
    
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
    
    total_samples = len(labels)
    
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
    
    # Overall row
    row = f"{'Total':<12}"
    for c in range(num_classes):
        count = int(np.sum(labels == c))
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
    parser = argparse.ArgumentParser(description='PPXFL Data Partitioning')
    parser.add_argument('--processed-dir', type=str, default=None,
                        help='Directory with preprocessed data (all_images.npy, all_labels.npy)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for client partitions')
    parser.add_argument('--num-clients', type=int, default=4,
                        help='Number of federated clients K (default: 4)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha (default: 0.5, lower = more non-IID)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()
    
    # Default paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.processed_dir is None:
        args.processed_dir = os.path.join(project_root, 'data', 'processed')
    if args.output is None:
        args.output = os.path.join(project_root, 'data', 'clients')
    
    # Load preprocessed data
    images_path = os.path.join(args.processed_dir, 'all_images.npy')
    labels_path = os.path.join(args.processed_dir, 'all_labels.npy')
    
    if not os.path.exists(images_path):
        print(f"[ERROR] {images_path} not found. Run preprocess.py first.")
        sys.exit(1)
    
    print("Loading preprocessed data...")
    images = np.load(images_path)
    labels = np.load(labels_path)
    print(f"  Loaded {len(images)} images, {len(labels)} labels")
    
    # Run Dirichlet partitioning
    print(f"\nPartitioning with Dirichlet(α={args.alpha}), K={args.num_clients} clients...")
    client_indices = dirichlet_partition(
        labels, 
        num_clients=args.num_clients, 
        alpha=args.alpha, 
        seed=args.seed
    )
    
    # Print statistics
    stats = compute_partition_stats(labels, client_indices)
    
    # Save client datasets
    print(f"\nSaving client datasets to {args.output}...")
    create_client_datasets(images, labels, client_indices, args.output)
    
    # Save partition metadata
    metadata = {
        'num_clients': args.num_clients,
        'alpha': args.alpha,
        'seed': args.seed,
        'total_samples': len(labels),
        'clients': stats
    }
    
    meta_path = os.path.join(args.output, 'partition_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved to {meta_path}")
    
    # Plot distribution
    fig_dir = os.path.join(project_root, 'results', 'figures')
    plot_partition_distribution(
        labels, client_indices,
        save_path=os.path.join(fig_dir, f'partition_alpha{args.alpha}.png')
    )
    
    # Save partition indices for reproducibility
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
