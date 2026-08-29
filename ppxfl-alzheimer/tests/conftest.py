"""Shared fixtures: synthetic stand-ins for the ADNI cohort.

No test in this suite reads real ADNI data. Every fixture builds a tiny
in-memory cohort (a handful of subjects, 8x8 images) that exercises the same
code paths as the full pipeline, so the suite runs on CPU in CI with no data
access agreement involved.
"""

import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture
def tiny_manifest():
    """15 subjects x 4 slices, three balanced classes, matching the real schema."""
    rows = []
    array_index = 0
    for label in range(3):
        for subject_number in range(5):
            subject_id = f'S{label}{subject_number:02d}'
            for slice_number in range(4):
                rows.append({
                    'subject_id': subject_id,
                    'label': label,
                    'array_index': array_index,
                    'slice': slice_number,
                })
                array_index += 1
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_manifest_path(tiny_manifest, tmp_path):
    path = tmp_path / 'manifest.csv'
    tiny_manifest.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def tiny_arrays(tiny_manifest):
    """Image/label arrays aligned with ``tiny_manifest`` by ``array_index``."""
    rng = np.random.RandomState(0)
    images = rng.rand(len(tiny_manifest), 8, 8).astype(np.float32)
    labels = tiny_manifest.sort_values('array_index')['label'].to_numpy()
    return images, labels


def make_state(**tensors):
    """Build a state dict from keyword tensors, preserving declaration order."""
    return OrderedDict((key, value) for key, value in tensors.items())


@pytest.fixture
def simple_state():
    """A two-parameter, one-buffer, one-counter state dict."""
    return make_state(
        weight=torch.zeros(2, 2),
        bias=torch.zeros(2),
        running_mean=torch.zeros(2),
        num_batches_tracked=torch.tensor(0, dtype=torch.long),
    )
