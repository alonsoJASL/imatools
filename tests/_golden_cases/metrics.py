"""Seed capture cases: pure ``ioutils`` metric/array math.

These exercise the full capture pipeline end to end (both ``json`` and ``npy``
serialization) and seed the contract for the future ``core/metrics.py`` migration.
The Wave-1 ``T1f · char-metrics`` task extends this module with the remaining metric
functions; T0 only needs it to prove the harness writes real golden files.

T1f additions: ``compare_large_arrays``, ``near`` (from ioutils), and
``compare_vector_field`` / ``compare_scalar_field`` (from compare_from_mapping).
The latter two are loaded without executing the module (argparse runs at top-level
there), so their bodies are reproduced verbatim here and called via local wrappers.
"""

from __future__ import annotations

import _fixtures as fx
import numpy as np
from _capture_golden import CaptureCase

from imatools.common import ioutils

# ---------------------------------------------------------------------------
# compare_from_mapping — load pure functions WITHOUT executing the module.
#
# compare_from_mapping.py runs ``argparse.ArgumentParser.parse_args()`` at module
# top level (line 47), which SystemExits if we ``import`` it. Instead we reproduce
# the two function bodies verbatim here (verified against master @ head, lines 13-36).
# ---------------------------------------------------------------------------


def compare_vector_field(v0, v1, r):
    """Verbatim copy of master's ``compare_from_mapping.compare_vector_field``."""
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
    """Verbatim copy of master's ``compare_from_mapping.compare_scalar_field``."""
    l2 = (s0 - s1) ** 2
    abs_diff = np.abs(s0 - s1)
    midic = {
        "diff_square": l2,
        "diff_abs": abs_diff,
        "s0": s0,
        "s1": s1,
    }
    return midic


# ---------------------------------------------------------------------------
# Fixture-derived constants shared across multiple cases
# ---------------------------------------------------------------------------

# Unit-normalised version of the shared vector field.  ``compare_vector_field``
# computes arccos(dot_product), which requires dot products in [-1, 1]; raw
# ``fx.vector_field()`` vectors are not unit-length.
_vf_raw = fx.vector_field()
_norms = np.linalg.norm(_vf_raw, axis=1, keepdims=True)
# Avoid division by zero for the all-zero first/last row.
_norms = np.where(_norms == 0, 1.0, _norms)
_UNIT_VECTOR_FIELD = _vf_raw / _norms

# Region tag array (integer, same length as the vector field).
_N = len(_UNIT_VECTOR_FIELD)
_REGION = np.arange(_N, dtype=int) % 3  # labels 0, 1, 2 cycling


CASES = [
    # ------------------------------------------------------------------
    # Seeded by T0 — do NOT modify these entries.
    # ------------------------------------------------------------------
    CaptureCase(
        name="metrics/performance_balanced",
        func=ioutils.performanceMetrics,
        args=(50, 40, 5, 5),
        fmt="json",
    ),
    CaptureCase(
        name="metrics/performance_degenerate_zeros",
        func=ioutils.performanceMetrics,
        args=(0, 0, 0, 0),
        fmt="json",
    ),
    CaptureCase(
        name="metrics/boxplot_linspace",
        func=ioutils.get_boxplot_values,
        args=(fx.scalar_field(),),
        fmt="json",
    ),
    CaptureCase(
        name="metrics/l2_norm_vector_field",
        func=ioutils.l2_norm,
        args=(fx.vector_field(),),
        fmt="npy",
    ),
    CaptureCase(
        name="metrics/classify_array_bins",
        func=ioutils.classify_array,
        args=(fx.scalar_field(), fx.classification_thresholds()),
        fmt="npy",
    ),
    CaptureCase(
        name="metrics/count_values_in_ranges",
        func=ioutils.count_values_in_ranges,
        args=(fx.scalar_field(), fx.classification_thresholds()),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # T1f additions
    # ------------------------------------------------------------------
    # near ---------------------------------------------------------------
    CaptureCase(
        name="metrics/near_equal",
        func=ioutils.near,
        args=(1.0, 1.0 + 1e-10),
        fmt="json",
    ),
    CaptureCase(
        name="metrics/near_far",
        func=ioutils.near,
        args=(1.0, 1.0 + 1e-6),
        fmt="json",
    ),
    # compare_large_arrays -----------------------------------------------
    # Returns a dict with ndarray values; ``_NumpyEncoder`` serializes them.
    CaptureCase(
        name="metrics/compare_large_arrays_basic",
        func=ioutils.compare_large_arrays,
        args=(fx.scalar_field(), fx.scalar_field()[::-1]),
        fmt="json",
    ),
    # Edge case: identical arrays -> diff_square = 0, cosine_similarity = 1
    CaptureCase(
        name="metrics/compare_large_arrays_identical",
        func=ioutils.compare_large_arrays,
        args=(fx.scalar_field(), fx.scalar_field()),
        fmt="json",
    ),
    # compare_scalar_field -----------------------------------------------
    CaptureCase(
        name="metrics/compare_scalar_field_linspace",
        func=compare_scalar_field,
        args=(fx.scalar_field(), fx.scalar_field()[::-1]),
        fmt="json",
    ),
    # Edge case: identical fields -> all diffs zero
    CaptureCase(
        name="metrics/compare_scalar_field_identical",
        func=compare_scalar_field,
        args=(fx.scalar_field(), fx.scalar_field()),
        fmt="json",
    ),
    # compare_vector_field -----------------------------------------------
    # Uses unit-normalised vectors so arccos receives values in [-1, 1].
    CaptureCase(
        name="metrics/compare_vector_field_unit",
        func=compare_vector_field,
        args=(_UNIT_VECTOR_FIELD, _UNIT_VECTOR_FIELD[::-1], _REGION),
        fmt="json",
    ),
    # Edge case: identical unit vectors -> dot_product = 1, angle = 0
    CaptureCase(
        name="metrics/compare_vector_field_identical",
        func=compare_vector_field,
        args=(_UNIT_VECTOR_FIELD, _UNIT_VECTOR_FIELD, _REGION),
        fmt="json",
    ),
]
