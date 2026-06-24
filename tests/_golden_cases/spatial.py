"""Capture cases for ``imatools.core.spatial`` (T1c).

Functions covered:
- ``create_normal_vector_for_plane(axis, angle)`` — pure math, returns ndarray.
- ``set_direction_as(im, ref)`` — copies direction from ref to im, returns sitk.Image.
- ``fix_header_to_axis_aligned(hdr)`` — normalises NRRD header dict, returns dict.

Functions NOT captured (broken in master):
- ``create_image_at_plane(image, point_on_plane, axis, angle)``
- ``create_image_at_plane_from_vector(image, point_on_plane, normal_vector)``

Both call ``sitk.AffineTransform.SetMatrix(normal_vector + [0, 0, 1])`` where
``normal_vector`` is a (3,) numpy array, yielding a 3-element operand that SimpleITK
rejects (it expects 9 elements for a 3x3 matrix).  Master always raises::

    sitk::ERROR: Length of input (3) does not match matrix dimensions (3, 3)

Since master is the oracle and it throws for every input, no golden can be captured.
Tests for these two functions are authored as xfail stubs in ``test_core_spatial.py``
with a note for the migration author (T2a3) to fix the body.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from _capture_golden import CaptureCase

from imatools.common import itktools

# ---------------------------------------------------------------------------
# Fixtures built inline — no shared fixture dependency required for spatial
# ---------------------------------------------------------------------------

# Small label image (12x12x12, spacing (1, 1, 2), origin (0, 0, 0))
_ARR = np.zeros((12, 12, 12), dtype=np.uint8)
_ARR[2:6, 2:6, 2:6] = 1
_ARR[6:10, 6:10, 6:10] = 2
_ARR[3:5, 7:9, 3:5] = 3

_DEFAULT_SPACING = (1.0, 1.0, 2.0)
_DEFAULT_ORIGIN = (0.0, 0.0, 0.0)


def _make_label_image():
    img = sitk.GetImageFromArray(_ARR.copy())
    img.SetSpacing(_DEFAULT_SPACING)
    img.SetOrigin(_DEFAULT_ORIGIN)
    return img


# A second image with a different direction and spacing used to test set_direction_as.
def _make_ref_image():
    arr = np.zeros((12, 12, 12), dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 2.0))
    img.SetOrigin((1.0, 1.0, 1.0))
    # Rotate 90 degrees around z: direction matrix (col-major flat) for such a rotation
    # [0, -1, 0,  1, 0, 0,  0, 0, 1] — this is a valid orthogonal direction cosine matrix
    img.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    return img


# ---------------------------------------------------------------------------
# Inline header for fix_header_to_axis_aligned (off-diagonal input)
# ---------------------------------------------------------------------------

_HDR_OBLIQUE = {
    "space directions": np.array([[0.8, 0.1, 0.0], [0.0, 0.9, 0.05], [0.0, 0.0, 1.1]], dtype=float),
    "space origin": np.array([1.0, 2.0, 3.0]),
}

# Second case: already axis-aligned (diag 1, 1, 1 with float noise = 0)
_HDR_ALIGNED = {
    "space directions": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
    "space origin": np.array([0.0, 0.0, 0.0]),
}


# ---------------------------------------------------------------------------
# Reduce helpers
# ---------------------------------------------------------------------------


def _reduce_fixed_header(hdr: dict) -> dict:
    """Convert the output of fix_header_to_axis_aligned to a JSON-serialisable dict.

    The returned header has ``space directions`` as an ndarray → convert to nested
    list; ``space origin`` similarly; ``srow_*`` are already strings.
    """
    return {
        "space directions": np.asarray(hdr["space directions"]).tolist(),
        "space origin": np.asarray(hdr["space origin"]).tolist(),
        "srow_x": hdr["srow_x"],
        "srow_y": hdr["srow_y"],
        "srow_z": hdr["srow_z"],
    }


def _reduce_sitk_image_to_direction_dict(im: sitk.Image) -> dict:
    """Return the header metadata of *im* as a JSON-serialisable dict.

    Used to characterise ``set_direction_as``, whose sole visible side-effect is that
    the returned image carries the direction from the reference.
    """
    return {
        "direction": list(im.GetDirection()),
        "spacing": list(im.GetSpacing()),
        "origin": list(im.GetOrigin()),
    }


# ---------------------------------------------------------------------------
# CASES
# ---------------------------------------------------------------------------

CASES = [
    # ------------------------------------------------------------------
    # create_normal_vector_for_plane — pure rotation math
    # ------------------------------------------------------------------
    # Note: the function computes ``angle_rad = np.radians(angle)`` but then uses
    # ``angle`` directly in the rotation matrix (angle_rad is unused).  The ``angle``
    # parameter is therefore treated as radians, not degrees, by the actual code.
    # We characterise this exact master behaviour (even though it looks like a bug).
    CaptureCase(
        name="spatial/normal_vector_z_zero",
        func=itktools.create_normal_vector_for_plane,
        args=("z", 0.0),
        fmt="npy",
    ),
    CaptureCase(
        name="spatial/normal_vector_z_quarter_pi",
        func=itktools.create_normal_vector_for_plane,
        args=("z", np.pi / 4),
        fmt="npy",
    ),
    CaptureCase(
        name="spatial/normal_vector_x_half_pi",
        func=itktools.create_normal_vector_for_plane,
        args=("x", np.pi / 2),
        fmt="npy",
    ),
    CaptureCase(
        name="spatial/normal_vector_y_pi",
        func=itktools.create_normal_vector_for_plane,
        args=("y", np.pi),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # set_direction_as — copies direction from ref image to im
    # ------------------------------------------------------------------
    CaptureCase(
        name="spatial/set_direction_as_basic",
        func=itktools.set_direction_as,
        args=(_make_label_image(), _make_ref_image()),
        reduce=_reduce_sitk_image_to_direction_dict,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # fix_header_to_axis_aligned — NRRD header normalisation
    # ------------------------------------------------------------------
    CaptureCase(
        name="spatial/fix_header_oblique",
        func=itktools.fix_header_to_axis_aligned,
        args=(_HDR_OBLIQUE,),
        reduce=_reduce_fixed_header,
        fmt="json",
    ),
    CaptureCase(
        name="spatial/fix_header_aligned",
        func=itktools.fix_header_to_axis_aligned,
        args=(_HDR_ALIGNED,),
        reduce=_reduce_fixed_header,
        fmt="json",
    ),
]
