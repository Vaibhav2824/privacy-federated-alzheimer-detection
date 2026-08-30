"""
xai_v3.py — region attribution, and what privacy does to it.

Explanations in this project live in a common anatomical space: every subject
is registered to MNI152 and every feature is a named Harvard-Oxford region, so
an attribution is a value per region and two models' attributions are directly
comparable.  That is what the earlier slice-space pipeline could not support —
without registration, a heat map at pixel (i, j) refers to different anatomy in
different subjects and cannot be averaged or compared at all.

Three quantities are reported:

``agreement``
    Spearman correlation between a model's region importance and the
    non-private centralised reference.  It answers whether a federated or
    private model reaches its decision by looking at the same anatomy, which
    accuracy alone cannot tell you.

``medial_temporal_share``
    The fraction of total attribution mass falling in hippocampus and amygdala.
    Structural change there is the best-established imaging correlate of
    Alzheimer's, so this is a directional check against known anatomy rather
    than a self-consistency check.

``top_k_overlap``
    Jaccard overlap of the most-attributed regions, which is what a clinician
    actually reads off an explanation.

Nothing here validates a model clinically.  Agreement with a reference model or
with expected anatomy is evidence about the model, not about a patient.
"""

from __future__ import annotations

import collections

import numpy as np

from .features import MEDIAL_TEMPORAL_REGIONS


def region_importance(weights: np.ndarray, feature_names: list[str]) -> dict:
    """Attribution per anatomical region, summed over its features and classes.

    ``weights`` is ``(n_features, n_classes)``.  Magnitudes are summed because
    the question is which anatomy the decision rests on, not the direction in
    which each class is pushed.
    """
    magnitude = np.abs(np.asarray(weights)).sum(axis=1)
    totals: dict[str, float] = collections.defaultdict(float)
    for value, name in zip(magnitude, feature_names):
        region = name.split("::", 1)[-1]
        totals[region] += float(value)
    return dict(totals)


def normalise(importance: dict) -> dict:
    total = sum(importance.values())
    if total <= 0:
        return dict.fromkeys(importance, 0.0)
    return {k: v / total for k, v in importance.items()}


def agreement(reference: dict, candidate: dict) -> float:
    """Spearman correlation between two region-importance maps."""
    from scipy.stats import spearmanr

    regions = sorted(set(reference) & set(candidate))
    if len(regions) < 3:
        return float("nan")
    a = [reference[r] for r in regions]
    b = [candidate[r] for r in regions]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def medial_temporal_share(importance: dict) -> float:
    """Share of attribution mass in hippocampus and amygdala."""
    normalised = normalise(importance)
    return float(sum(normalised.get(r, 0.0) for r in MEDIAL_TEMPORAL_REGIONS))


def top_k_overlap(reference: dict, candidate: dict, k: int = 10) -> float:
    """Jaccard overlap of the top-``k`` regions of two attribution maps."""
    def top(mapping):
        return {r for r, _ in sorted(mapping.items(), key=lambda kv: -kv[1])[:k]}

    a, b = top(reference), top(candidate)
    union = a | b
    return float(len(a & b) / len(union)) if union else float("nan")


def compare(reference_weights, candidate_weights, feature_names, k: int = 10) -> dict:
    reference = region_importance(reference_weights, feature_names)
    candidate = region_importance(candidate_weights, feature_names)
    return {
        "agreement_spearman": agreement(reference, candidate),
        "top_k_overlap": top_k_overlap(reference, candidate, k=k),
        "medial_temporal_share": medial_temporal_share(candidate),
        "reference_medial_temporal_share": medial_temporal_share(reference),
        "top_regions": [
            {"region": r, "share": round(v, 5)}
            for r, v in sorted(normalise(candidate).items(), key=lambda kv: -kv[1])[:k]
        ],
    }


def attribution_map(weights: np.ndarray, feature_names: list[str], label_volume,
                    region_names: list[str]) -> np.ndarray:
    """Paint region attributions back onto the MNI grid for figures."""
    importance = normalise(region_importance(weights, feature_names))
    volume = np.zeros(label_volume.shape, dtype=np.float32)
    for index, name in enumerate(region_names, start=1):
        share = importance.get(name)
        if share:
            volume[label_volume == index] = share
    return volume
