"""Characterization tests for ``imatools.core.scar`` (M1.6a).

Functions are imported from their TARGET homes; tests are expected to PASS now
that the migration is in place.

The golden values were captured from master's ``ScarQuantificationTools`` methods
and the inline kernel in ``enhance_debug_scar.py`` using the same synthetic fixtures.

``enhance_scar_array`` uses the same seeded numpy arrays as the capture cases so
the inputs are bit-identical to what master saw.

``mask_segmentation_above_threshold`` preserves master's arg-order bug verbatim
(documented in ``core/scar.py`); its golden encodes the buggy behaviour.
``get_mask_with_restrictions`` (straggler, M1.6a) is tested transitively through
the scar tests and via its own identity-shim smoke test below.
"""

from __future__ import annotations

import _fixtures as fx
import numpy as np
import pytest
import SimpleITK as sitk

from imatools.core.image import get_mask_with_restrictions
from imatools.core.scar import (
    enhance_scar_array,
    get_scar_method,
    get_threshold,
    get_threshold_values,
    mask_voxels_above_threshold,
)

# ---------------------------------------------------------------------------
# Constants shared with capture cases (must be byte-identical)
# ---------------------------------------------------------------------------

_MEAN_BP = 152.3
_STD_BP = 31.7
_THRESHOLDS_IIR = [0.97, 1.2, 1.32]
_THRESHOLDS_MSD = [0.97, 1.2, 1.32]

np.random.seed(42)
_SHAPE = (5, 5, 5)
_SCAR_ARR = np.zeros(_SHAPE, dtype=np.int32)
_SCAR_ARR[2, 2, 2] = 2
_SCAR_ARR[1, 1, 1] = 1
_IM_ARR = (np.random.rand(*_SHAPE) * 300).astype(np.float64)


# ---------------------------------------------------------------------------
# get_scar_method
# ---------------------------------------------------------------------------


def test_get_scar_method_iir(golden):
    expected = golden("scar/get_scar_method_iir")
    assert get_scar_method("iir") == expected


def test_get_scar_method_msd(golden):
    expected = golden("scar/get_scar_method_msd")
    assert get_scar_method("msd") == expected


def test_get_scar_method_invalid():
    with pytest.raises(KeyError):
        get_scar_method("unknown")


# ---------------------------------------------------------------------------
# get_threshold
# ---------------------------------------------------------------------------


def test_get_threshold_iir_positive(golden):
    expected = golden("scar/get_threshold_iir_positive")
    result = get_threshold(1, 1.2, _MEAN_BP, _STD_BP)
    assert abs(result - expected) < 1e-9


def test_get_threshold_iir_zero_value(golden):
    expected = golden("scar/get_threshold_iir_zero_value")
    result = get_threshold(1, 0.0, _MEAN_BP, _STD_BP)
    assert result == expected


def test_get_threshold_msd_positive(golden):
    expected = golden("scar/get_threshold_msd_positive")
    result = get_threshold(2, 1.2, _MEAN_BP, _STD_BP)
    assert abs(result - expected) < 1e-9


def test_get_threshold_msd_zero_value(golden):
    expected = golden("scar/get_threshold_msd_zero_value")
    result = get_threshold(2, 0.0, _MEAN_BP, _STD_BP)
    assert result == expected


# ---------------------------------------------------------------------------
# get_threshold_values
# ---------------------------------------------------------------------------


def test_get_threshold_values_iir(golden):
    expected = golden("scar/get_threshold_values_iir")
    result = get_threshold_values(_THRESHOLDS_IIR, _MEAN_BP, _STD_BP, "iir")
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_get_threshold_values_msd(golden):
    expected = golden("scar/get_threshold_values_msd")
    result = get_threshold_values(_THRESHOLDS_MSD, _MEAN_BP, _STD_BP, "msd")
    np.testing.assert_allclose(result, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# enhance_scar_array
# ---------------------------------------------------------------------------


def test_enhance_scar_array_iir(golden):
    threshold_values = get_threshold_values(_THRESHOLDS_IIR, _MEAN_BP, _STD_BP, "iir")
    result = enhance_scar_array(_SCAR_ARR, _IM_ARR, threshold_values)
    expected = golden("scar/enhance_scar_array_iir")
    np.testing.assert_array_equal(result, expected)


def test_enhance_scar_array_msd(golden):
    threshold_values = get_threshold_values(_THRESHOLDS_MSD, _MEAN_BP, _STD_BP, "msd")
    result = enhance_scar_array(_SCAR_ARR, _IM_ARR, threshold_values)
    expected = golden("scar/enhance_scar_array_msd")
    np.testing.assert_array_equal(result, expected)


def test_enhance_scar_array_corridor_unchanged():
    """Voxels with scar_value <= 1 must not be changed."""
    threshold_values = [100.0]
    scar = np.array([[[0, 1, 2]]], dtype=np.int32)
    im = np.array([[[50.0, 50.0, 200.0]]])
    result = enhance_scar_array(scar, im, threshold_values)
    assert result[0, 0, 0] == 0  # background unchanged
    assert result[0, 0, 1] == 1  # corridor unchanged
    assert result[0, 0, 2] == 3  # scar: 2 + 1 threshold exceeded


# ---------------------------------------------------------------------------
# mask_voxels_above_threshold (smoke test — no golden, depends on mask_image)
# ---------------------------------------------------------------------------


def test_mask_voxels_above_threshold_smoke():
    """Smoke test: returns a SimpleITK image with correct size."""
    im = fx.label_image()
    mask = fx.binary_image()
    result = mask_voxels_above_threshold(
        im,
        mask,
        thres_mean=_MEAN_BP,
        thres_std=_STD_BP,
        scar_method="iir",
        thres_value=0.0,
        mask_value=0,
    )
    assert isinstance(result, sitk.Image)
    assert result.GetSize() == im.GetSize()


# ---------------------------------------------------------------------------
# get_mask_with_restrictions — straggler identity + behaviour smoke test
# ---------------------------------------------------------------------------


def test_get_mask_with_restrictions_identity_shim():
    """Object-identity shim: itktools.get_mask_with_restrictions IS core.image version."""
    from imatools.common.itktools import get_mask_with_restrictions as itk_fn
    from imatools.core.image import get_mask_with_restrictions as core_fn

    assert itk_fn is core_fn


def test_get_mask_with_restrictions_smoke():
    """Smoke: returns a binary mask image with the correct size."""
    im = fx.label_image()
    mask = fx.binary_image()
    result = get_mask_with_restrictions(im, mask, threshold=0)
    assert isinstance(result, sitk.Image)
    assert result.GetSize() == im.GetSize()
