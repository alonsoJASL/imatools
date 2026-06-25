# src/imatools/core/metrics.py
"""Metric and comparison functions migrated from ``imatools.common.ioutils``
and copied from ``imatools.compare_from_mapping`` (T2c1).

The 7 functions moved from ``ioutils`` are the authoritative implementations;
the old ``imatools.common.ioutils`` module re-exports them via a shim at its
bottom.

The 2 functions copied from ``compare_from_mapping`` (``compare_vector_field``,
``compare_scalar_field``) live here as the canonical target-layer copies.
``compare_from_mapping.py`` is a deferred CLI script that keeps its own copies;
it is NOT shimmed.

No lazy accessor needed — all function bodies use only ``numpy`` and
``scipy.spatial.distance``, with no calls back into ``ioutils``.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Moved from imatools.common.ioutils (verbatim)
# ---------------------------------------------------------------------------


def l2_norm(a):
    return np.linalg.norm(a, axis=1)


def near(value1, value2, tol=1e-8):
    return np.abs(value1 - value2) <= tol


def performanceMetrics(tp, tn, fp, fn):  # noqa: N802
    den_jaccard = tp + fn + fp
    den_precision = tp + fp
    den_recall = tp + fn
    den_accuracy = tp + tn + fp + fn
    den_dice = 2 * tp + fp + fn

    jaccard = tp / (tp + fn + fp) if den_jaccard > 0 else np.nan
    precision = tp / (tp + fp) if den_precision > 0 else np.nan
    recall = tp / (tp + fn) if den_recall > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if den_accuracy > 0 else np.nan
    dice = (2 * tp) / (2 * tp + fp + fn) if den_dice > 0 else np.nan

    out_dic = {
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "dice": dice,
    }

    return out_dic


def get_boxplot_values(data, whisker=1.5):
    low_quartile = np.nanpercentile(data, 25, method="nearest")
    high_quartile = np.nanpercentile(data, 75, method="nearest")
    iqr = high_quartile - low_quartile
    low_whis = low_quartile - whisker * iqr
    high_whis = high_quartile + whisker * iqr
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    midic = {
        "min": min_val,
        "low_whisker": np.max([low_whis, min_val]),
        "low_quartile": low_quartile,
        "median": np.nanmedian(data),
        "high_quartile": high_quartile,
        "high_whisker": np.min([high_whis, max_val]),
        "max": max_val,
    }
    return midic


def compare_large_arrays(s0, s1, name0="s0", name1="s1"):
    from scipy.spatial.distance import cosine  # noqa: PLC0415

    l2 = (s0 - s1) ** 2
    abs_diff = np.abs(s0 - s1)
    cosine_similarity = 1 - cosine(s0, s1)
    midic = {
        "diff_square": l2,
        "diff_abs": abs_diff,
        "cosine_similarity": cosine_similarity,
        name0: s0,
        name1: s1,
    }

    return midic


def classify_array(arr, thresholds: list):
    """
    Classifies an array based on given thresholds.

    Parameters:
    arr (numpy.ndarray): Array of values to classify.
    thresholds (list): List of tuples where each tuple contains the name of the classification and the corresponding threshold values.

    Returns:
    numpy.ndarray: Array of classifications.
    """
    classifications = np.zeros_like(arr, dtype=int)
    for i, value in enumerate(arr):
        for j, threshold in enumerate(thresholds):
            if threshold[0] <= value < threshold[1]:
                classifications[i] = j
                break
    return classifications


def count_values_in_ranges(arr, thresholds: list) -> dict:
    """
    Counts the number of values in each range.

    Parameters:
    arr (numpy.ndarray): Array of values to classify.
    thresholds (list): List of tuples where each tuple contains the name of the classification and the corresponding threshold values.

    Returns:
    dict: Dictionary containing the number of values in each range.
    """
    classifications = classify_array(arr, thresholds)
    counts = {}
    for ix in range(len(thresholds)):
        counts[ix] = np.count_nonzero(classifications == ix)

    return counts


# ---------------------------------------------------------------------------
# Copied from imatools.compare_from_mapping (verbatim — NOT a re-export shim;
# compare_from_mapping.py is a deferred CLI script and must NOT be imported).
# ---------------------------------------------------------------------------


def compare_vector_field(v0, v1, r):
    dotp = np.sum(np.multiply(v0, v1), axis=1)
    abs_dotp = np.abs(dotp)
    midic = {
        "region": r,
        "dot_product": dotp,
        "angle": np.arccos(dotp),
        "abs_dot_product": abs_dotp,
        "angle_from_absdot": np.arccos(abs_dotp),
    }

    return midic


def compare_scalar_field(s0, s1):
    l2 = (s0 - s1) ** 2
    abs_diff = np.abs(s0 - s1)
    midic = {
        "diff_square": l2,
        "diff_abs": abs_diff,
        "s0": s0,
        "s1": s1,
    }

    return midic
