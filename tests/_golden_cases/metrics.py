"""Seed capture cases: pure ``ioutils`` metric/array math.

These exercise the full capture pipeline end to end (both ``json`` and ``npy``
serialization) and seed the contract for the future ``core/metrics.py`` migration.
The Wave-1 ``T1f · char-metrics`` task extends this module with the remaining metric
functions; T0 only needs it to prove the harness writes real golden files.
"""

from __future__ import annotations

import _fixtures as fx
from _capture_golden import CaptureCase

from imatools.common import ioutils

CASES = [
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
]
