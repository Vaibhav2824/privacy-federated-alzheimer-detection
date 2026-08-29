"""
preprocess.py — MRI Preprocessing Pipeline
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Loads NIfTI MRI scans from ADNI folders, extracts middle axial slices,
normalises intensity, resizes to 224×224, and saves as .npy arrays.
"""

import argparse
import csv
import os
import re
import sys
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

SUBJECT_ID_RE = re.compile(r'\d{3}_S_\d{4}')


def find_nifti_files(data_dir):
    """Recursively find all .nii and .nii.gz files in a directory."""
    nifti_files = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                nifti_files.append(os.path.join(root, f))
    return sorted(nifti_files)


def extract_subject_id(nifti_path):
    """Extract the true ADNI subject ID (e.g. 002_S_0619) from the scan path.

    The subject ID is a directory component, not part of the scan filename —
    a single subject has many scans, so parsing it from the filename (as the
    original pipeline did) silently treats every scan as its own subject.
    """
    match = SUBJECT_ID_RE.search(nifti_path)
    if match is None:
        raise ValueError(f"Could not find ADNI subject ID (NNN_S_NNNN) in path: {nifti_path}")
    return match.group(0)


def extract_scan_id(nifti_path):
    """Extract the ADNI image/scan UID (e.g. I12345) from the filename, if present."""
    stem = os.path.basename(nifti_path).split('.')[0]
    match = re.search(r'I\d+', stem)
    return match.group(0) if match else stem


def load_and_process_nifti(nifti_path, target_size=(224, 224), num_slices=3):
    """
    Load NIfTI file, extract key axial slices, normalise, and resize.

    Args:
        nifti_path: Path to .nii file
        target_size: Output image size (H, W)
        num_slices: Number of slices to extract around the middle

    Returns:
        processed_slices: list of numpy arrays (H, W) normalised to [0, 1]
    """
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()

        # Handle different orientations - ensure we get axial view
        # Data shape is typically (sagittal, coronal, axial) or similar
        if len(data.shape) == 4:
            data = data[:, :, :, 0]  # Take first volume if 4D

        # Extract slices along the axis with most slices (typically axial)
        # Find the axis with the most elements for axial slices
        axial_axis = np.argmax(data.shape)
        n_slices = data.shape[axial_axis]

        # Get the middle slice and surrounding slices
        mid = n_slices // 2
        slice_indices = [mid - n_slices // 6, mid, mid + n_slices // 6]

        processed_slices = []
        for idx in slice_indices:
            idx = max(0, min(idx, n_slices - 1))

            if axial_axis == 0:
                slice_2d = data[idx, :, :]
            elif axial_axis == 1:
                slice_2d = data[:, idx, :]
            else:
                slice_2d = data[:, :, idx]

            # Intensity normalisation (z-score then clip to [0, 1])
            if slice_2d.std() > 0:
                slice_2d = (slice_2d - slice_2d.mean()) / slice_2d.std()
                # Clip to reasonable range and scale to [0,1]
                slice_2d = np.clip(slice_2d, -3, 3)
                slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
            else:
                slice_2d = np.zeros_like(slice_2d)

            # Resize to target size using PIL
            pil_img = Image.fromarray((slice_2d * 255).astype(np.uint8))
            pil_img = pil_img.resize(target_size, Image.LANCZOS)
            processed = np.array(pil_img).astype(np.float32) / 255.0

            processed_slices.append(processed)

        return processed_slices

    except Exception as e:
        print(f"  [ERROR] Failed to process {nifti_path}: {e}")
        return None


def preprocess_dataset(raw_dirs, output_dir, target_size=(224, 224)):
    """
    Process all MRI scans from class-labelled directories.

    Args:
        raw_dirs: Dict mapping class label -> directory path
                  e.g., {'AD': 'AD-150', 'MCI': 'MCI-150', 'CN': 'CN-150'}
        output_dir: Path to save processed .npy files
        target_size: Output image size

    Returns:
        stats: Dict with class counts and processing statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    stats = {}
    all_images = []
    all_labels = []
    manifest_rows = []
    label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
    array_index = 0

    for class_name, class_dir in raw_dirs.items():
        class_output = os.path.join(output_dir, class_name)
        os.makedirs(class_output, exist_ok=True)

        nifti_files = find_nifti_files(class_dir)
        print(f"\n{'='*60}")
        print(f"Processing {class_name}: {len(nifti_files)} NIfTI files found")
        print(f"{'='*60}")

        processed_count = 0
        failed_count = 0
        subjects_seen = set()

        for nifti_path in tqdm(nifti_files, desc=f"  {class_name}"):
            slices = load_and_process_nifti(nifti_path, target_size)

            if slices is not None:
                subject_id = extract_subject_id(nifti_path)
                scan_id = extract_scan_id(nifti_path)
                subjects_seen.add(subject_id)

                for i, s in enumerate(slices):
                    fname = f"{subject_id}_{scan_id}_slice{i}.npy"
                    np.save(os.path.join(class_output, fname), s)
                    all_images.append(s)
                    all_labels.append(label_map[class_name])
                    manifest_rows.append({
                        'array_index': array_index,
                        'subject_id': subject_id,
                        'scan_id': scan_id,
                        'class': class_name,
                        'label': label_map[class_name],
                        'slice_pos': i,
                        'source_path': nifti_path,
                    })
                    array_index += 1

                processed_count += 1
            else:
                failed_count += 1

        stats[class_name] = {
            'total_nifti': len(nifti_files),
            'unique_subjects': len(subjects_seen),
            'processed': processed_count,
            'failed': failed_count,
            'output_slices': processed_count * 3  # 3 slices per scan
        }
        print(f"  ✓ {processed_count} scans processed ({len(subjects_seen)} unique subjects), {failed_count} failed")

    # Save combined dataset
    all_images = np.array(all_images, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    np.save(os.path.join(output_dir, 'all_images.npy'), all_images)
    np.save(os.path.join(output_dir, 'all_labels.npy'), all_labels)

    manifest_path = os.path.join(output_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'array_index', 'subject_id', 'scan_id', 'class', 'label', 'slice_pos', 'source_path'
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"  ✓ Manifest saved to {manifest_path} ({len(manifest_rows)} rows)")

    total_unique_subjects = len({row['subject_id'] for row in manifest_rows})
    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}")
    print(f"  Total images: {len(all_images)}")
    print(f"  Shape: {all_images.shape}")
    print(f"  Total unique subjects: {total_unique_subjects}")
    print(f"  Labels distribution: CN={np.sum(all_labels==0)}, MCI={np.sum(all_labels==1)}, AD={np.sum(all_labels==2)}")

    return stats, all_images, all_labels


def visualise_samples(output_dir, save_path=None):
    """Generate a 3×3 grid of sample preprocessed MRI slices."""
    class_names = ['CN', 'MCI', 'AD']
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Preprocessed MRI Samples (224×224)', fontsize=16, fontweight='bold')

    for row, class_name in enumerate(class_names):
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        npy_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])

        for col in range(3):
            if col < len(npy_files):
                img = np.load(os.path.join(class_dir, npy_files[col]))
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].set_title(f'{class_name} — Sample {col+1}', fontsize=11)
            axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Sample grid saved to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PPXFL MRI Preprocessing Pipeline')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),),
                        help='Root directory containing AD-150, MCI-150, CN-150 folders')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for processed data')
    parser.add_argument('--target-size', type=int, default=224,
                        help='Target image size (default: 224)')
    args = parser.parse_args()

    # Set paths
    data_root = args.data_root
    if args.output is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
    else:
        output_dir = args.output

    # Define class directories
    raw_dirs = {
        'AD': os.path.join(data_root, 'AD-150'),
        'MCI': os.path.join(data_root, 'MCI-150'),
        'CN': os.path.join(data_root, 'CN-150'),
    }

    # Verify directories exist
    for class_name, class_dir in raw_dirs.items():
        if not os.path.exists(class_dir):
            print(f"[ERROR] Directory not found: {class_dir}")
            sys.exit(1)
        print(f"  ✓ Found {class_name} directory: {class_dir}")

    target_size = (args.target_size, args.target_size)

    # Run preprocessing
    print("\nStarting preprocessing pipeline...")
    print(f"  Output: {output_dir}")
    print(f"  Target size: {target_size}")

    stats, images, labels = preprocess_dataset(raw_dirs, output_dir, target_size)

    # Generate sample visualisation
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'figures')
    visualise_samples(output_dir, save_path=os.path.join(fig_dir, 'preprocessing_samples.png'))

    # Save stats
    import json
    stats_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'metrics', 'preprocessing_stats.json')
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Stats saved to {stats_path}")

    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
