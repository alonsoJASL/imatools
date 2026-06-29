"""Capture cases for scar quantification pure functions (M1.6a).

Functions captured from master's ``imatools.common.scarqtools.ScarQuantificationTools``
and ``imatools.enhance_debug_scar``.

``enhance_scar_array`` body is copied verbatim from ``enhance_debug_scar.py::main``
(the inline triple-loop kernel) — same pattern as the Wave-1 ``compare_vector_field``
copy — because the kernel is not an importable function in master.

``get_threshold_values`` is byte-identical in both ``enhance_debug_scar.py`` and
``pool_enhance_debug_scar.py``; captured once here.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from _capture_golden import CaptureCase

from imatools.common.scarqtools import ScarQuantificationTools

# ---------------------------------------------------------------------------
# Helpers: instantiate a default ScarQuantificationTools for method delegation.
# ---------------------------------------------------------------------------

_sqt_iir = ScarQuantificationTools(scar_method="iir")
_sqt_msd = ScarQuantificationTools(scar_method="msd")

# ---------------------------------------------------------------------------
# enhance_debug_scar -- the pure voxel-enhance kernel copied verbatim.
#
# The kernel lives inline in ``enhance_debug_scar.py::main``; it is NOT a standalone
# function in master. We reproduce the exact arithmetic here (lines 64–77 of
# enhance_debug_scar.py) and wrap it in a callable for capture.
# The consolidated ``get_threshold_values`` is imported from enhance_debug_scar
# (it is a proper top-level function there).
# ---------------------------------------------------------------------------


def _enhance_scar_array_verbatim(scar_array, im_array, threshold_values):
    """Verbatim copy of the triple-loop kernel from ``enhance_debug_scar.py::main``."""
    enhanced_array = np.copy(scar_array)
    for x in range(scar_array.shape[0]):
        for y in range(scar_array.shape[1]):
            for z in range(scar_array.shape[2]):
                scar_value = scar_array[x, y, z]
                lge_value = im_array[x, y, z]
                if scar_value > 1:
                    enhanced_value = 2
                    for th in threshold_values:
                        enhanced_value += 1 if lge_value > th else 0
                    enhanced_array[x, y, z] = enhanced_value
    return enhanced_array


# Under the capture env PYTHONPATH includes master/imatools, so bare import works.
from enhance_debug_scar import get_threshold_values as _get_threshold_values_master  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic fixtures for scar capture
# ---------------------------------------------------------------------------

# Numeric args for get_threshold / get_threshold_values
_MEAN_BP = 152.3
_STD_BP = 31.7
_THRESHOLDS_IIR = [0.97, 1.2, 1.32]
_THRESHOLDS_MSD = [0.97, 1.2, 1.32]

# Small 3-D scar + LGE arrays (5×5×5)
np.random.seed(42)
_SHAPE = (5, 5, 5)
# scar_array: background (0), corridor (1), scar-corridor (2)
_SCAR_ARR = np.zeros(_SHAPE, dtype=np.int32)
_SCAR_ARR[2, 2, 2] = 2  # single scar voxel
_SCAR_ARR[1, 1, 1] = 1  # corridor voxel (should not be enhanced)
_IM_ARR = (np.random.rand(*_SHAPE) * 300).astype(np.float64)  # synthetic LGE intensities

# Threshold values for iir method with the synthetic stats above
_THRES_VALUES_IIR = _get_threshold_values_master(_THRESHOLDS_IIR, _MEAN_BP, _STD_BP, "iir")
_THRES_VALUES_MSD = _get_threshold_values_master(_THRESHOLDS_MSD, _MEAN_BP, _STD_BP, "msd")

# ---------------------------------------------------------------------------
# prodStats.txt synthetic fixture
# ---------------------------------------------------------------------------

_PROD_STATS_CONTENT = (
    "IIR_method\n"
    f"{_MEAN_BP}\n"
    f"{_STD_BP}\n"
    "V=0.970, SCORE=5.2\n"
    "V=1.200, SCORE=3.1\n"
    "V=1.320, SCORE=1.4\n"
)

_PROD_STATS_MSD_CONTENT = (
    "MSD_method\n"
    f"{_MEAN_BP}\n"
    f"{_STD_BP}\n"
    "V=0.970, SCORE=5.2\n"
    "V=1.200, SCORE=3.1\n"
    "V=1.320, SCORE=1.4\n"
)

# Write to temp files at module load time so they're available for capture cases.
_TMP_DIR = Path(tempfile.mkdtemp())
_PROD_STATS_PATH = _TMP_DIR / "prodStats_iir.txt"
_PROD_STATS_MSD_PATH = _TMP_DIR / "prodStats_msd.txt"
_PROD_STATS_PATH.write_text(_PROD_STATS_CONTENT, encoding="utf-8")
_PROD_STATS_MSD_PATH.write_text(_PROD_STATS_MSD_CONTENT, encoding="utf-8")

# ---------------------------------------------------------------------------
# create_scar_options_file: capture the written JSON dict
# ---------------------------------------------------------------------------


def _capture_create_scar_options_file(**kwargs):
    """Write options.json to a temp dir; return the parsed dict."""
    d = _TMP_DIR / "opts"
    d.mkdir(exist_ok=True)
    _sqt_iir.create_scar_options_file(str(d), **kwargs)
    opt_file = kwargs.get("opt_file", "options.json")
    with open(d / opt_file, "r") as f:
        return json.load(f)


def _capture_create_scar_options_file_legacy():
    return _capture_create_scar_options_file(legacy=True)


def _capture_create_scar_options_file_radius():
    return _capture_create_scar_options_file(radius=True)


def _capture_create_scar_options_file_msd():
    d = _TMP_DIR / "opts_msd"
    d.mkdir(exist_ok=True)
    _sqt_msd.create_scar_options_file(str(d), method=2, threshold_values=[0.5, 1.0])
    with open(d / "options.json", "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# read_stats_from_file / get_bloodpool_stats_from_file
# ---------------------------------------------------------------------------


def _read_stats_iir():
    method, mean_bp, std_bp, thresholds, scores = _sqt_iir.read_stats_from_file(
        str(_PROD_STATS_PATH)
    )
    return {
        "method": method,
        "mean_bp": mean_bp,
        "std_bp": std_bp,
        "thresholds": thresholds,
        "scores": scores,
    }


def _read_stats_msd():
    method, mean_bp, std_bp, thresholds, scores = _sqt_iir.read_stats_from_file(
        str(_PROD_STATS_MSD_PATH)
    )
    return {
        "method": method,
        "mean_bp": mean_bp,
        "std_bp": std_bp,
        "thresholds": thresholds,
        "scores": scores,
    }


def _bloodpool_stats_iir():
    mean_bp, std_bp = _sqt_iir.get_bloodpool_stats_from_file(str(_PROD_STATS_PATH))
    return {"mean_bp": mean_bp, "std_bp": std_bp}


# ---------------------------------------------------------------------------
# CASES
# ---------------------------------------------------------------------------

CASES = [
    # ------------------------------------------------------------------
    # get_threshold — pure numeric
    # ------------------------------------------------------------------
    CaptureCase(
        name="scar/get_threshold_iir_positive",
        func=_sqt_iir.get_threshold,
        args=(1, 1.2, _MEAN_BP, _STD_BP),
        fmt="json",
    ),
    CaptureCase(
        name="scar/get_threshold_iir_zero_value",
        func=_sqt_iir.get_threshold,
        args=(1, 0.0, _MEAN_BP, _STD_BP),
        fmt="json",
    ),
    CaptureCase(
        name="scar/get_threshold_msd_positive",
        func=_sqt_iir.get_threshold,
        args=(2, 1.2, _MEAN_BP, _STD_BP),
        fmt="json",
    ),
    CaptureCase(
        name="scar/get_threshold_msd_zero_value",
        func=_sqt_iir.get_threshold,
        args=(2, 0.0, _MEAN_BP, _STD_BP),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # get_scar_method
    # ------------------------------------------------------------------
    CaptureCase(
        name="scar/get_scar_method_iir",
        func=_sqt_iir.get_scar_method,
        args=(),
        fmt="json",
    ),
    CaptureCase(
        name="scar/get_scar_method_msd",
        func=_sqt_msd.get_scar_method,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # get_threshold_values (from enhance_debug_scar)
    # ------------------------------------------------------------------
    CaptureCase(
        name="scar/get_threshold_values_iir",
        func=_get_threshold_values_master,
        args=(_THRESHOLDS_IIR, _MEAN_BP, _STD_BP, "iir"),
        fmt="json",
    ),
    CaptureCase(
        name="scar/get_threshold_values_msd",
        func=_get_threshold_values_master,
        args=(_THRESHOLDS_MSD, _MEAN_BP, _STD_BP, "msd"),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # enhance_scar_array (verbatim kernel)
    # ------------------------------------------------------------------
    CaptureCase(
        name="scar/enhance_scar_array_iir",
        func=_enhance_scar_array_verbatim,
        args=(_SCAR_ARR, _IM_ARR, _THRES_VALUES_IIR),
        fmt="npy",
    ),
    CaptureCase(
        name="scar/enhance_scar_array_msd",
        func=_enhance_scar_array_verbatim,
        args=(_SCAR_ARR, _IM_ARR, _THRES_VALUES_MSD),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # read_stats_from_file / get_bloodpool_stats_from_file
    # ------------------------------------------------------------------
    CaptureCase(
        name="scar/read_stats_iir",
        func=_read_stats_iir,
        args=(),
        fmt="json",
    ),
    CaptureCase(
        name="scar/read_stats_msd",
        func=_read_stats_msd,
        args=(),
        fmt="json",
    ),
    CaptureCase(
        name="scar/bloodpool_stats_iir",
        func=_bloodpool_stats_iir,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # create_scar_options_file
    # ------------------------------------------------------------------
    CaptureCase(
        name="scar/create_scar_options_default",
        func=_capture_create_scar_options_file,
        args=(),
        fmt="json",
    ),
    CaptureCase(
        name="scar/create_scar_options_legacy",
        func=_capture_create_scar_options_file_legacy,
        args=(),
        fmt="json",
    ),
    CaptureCase(
        name="scar/create_scar_options_radius",
        func=_capture_create_scar_options_file_radius,
        args=(),
        fmt="json",
    ),
    CaptureCase(
        name="scar/create_scar_options_msd",
        func=_capture_create_scar_options_file_msd,
        args=(),
        fmt="json",
    ),
]
