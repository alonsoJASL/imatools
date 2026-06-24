"""Characterization tests for ``imatools.core.spatial`` (T1c).

All tests import from the TARGET location ``imatools.core.spatial``.  That module
does not exist yet — it will be created by migration task T2a3.  Until then every
test is marked ``xfail(strict=False)`` so it is collected but does not block CI.

Golden values were captured from master via::

    ~/opt/anaconda3/bin/conda run -n imatools env \\
        PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module spatial --out tests/golden

where ``M = ~/dev/python/imatools.worktrees/master``.

Master bug (documented for T2a3)
---------------------------------
``create_image_at_plane`` and ``create_image_at_plane_from_vector`` are broken in
master: the call ``sitk.AffineTransform.SetMatrix(normal_vector + [0, 0, 1])``
passes a 3-element array where SimpleITK requires 9 elements for a 3-D affine matrix.
Master always raises::

    sitk::ERROR: Length of input (3) does not match matrix dimensions (3, 3)

No golden files are captured for these two functions (capture would fail at runtime).
The corresponding tests below are ``xfail`` stubs that verify the import path and
document the expected error so the T2a3 migration author can fix the body.

Comparison helpers
------------------
* **npy** goldens  → numpy arrays; compared with ``np.testing.assert_allclose``.
* **json** goldens → Python dicts/lists; list values converted to arrays for
  numeric comparison, string values compared exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Fixture helpers (must be identical to _golden_cases/spatial.py)
# ---------------------------------------------------------------------------

_ARR = np.zeros((12, 12, 12), dtype=np.uint8)
_ARR[2:6, 2:6, 2:6] = 1
_ARR[6:10, 6:10, 6:10] = 2
_ARR[3:5, 7:9, 3:5] = 3

_DEFAULT_SPACING = (1.0, 1.0, 2.0)
_DEFAULT_ORIGIN = (0.0, 0.0, 0.0)


def _make_label_image() -> sitk.Image:
    img = sitk.GetImageFromArray(_ARR.copy())
    img.SetSpacing(_DEFAULT_SPACING)
    img.SetOrigin(_DEFAULT_ORIGIN)
    return img


def _make_ref_image() -> sitk.Image:
    arr = np.zeros((12, 12, 12), dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 2.0))
    img.SetOrigin((1.0, 1.0, 1.0))
    img.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    return img


_HDR_OBLIQUE = {
    "space directions": np.array([[0.8, 0.1, 0.0], [0.0, 0.9, 0.05], [0.0, 0.0, 1.1]], dtype=float),
    "space origin": np.array([1.0, 2.0, 3.0]),
}

_HDR_ALIGNED = {
    "space directions": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
    "space origin": np.array([0.0, 0.0, 0.0]),
}


# ---------------------------------------------------------------------------
# create_normal_vector_for_plane
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a3", strict=False)
def test_normal_vector_z_zero(golden):
    """Rotating the z-axis around z by 0 radians → [0, 0, 1]."""
    from imatools.core.spatial import create_normal_vector_for_plane

    result = create_normal_vector_for_plane("z", 0.0)
    expected = golden("spatial/normal_vector_z_zero")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


@pytest.mark.xfail(reason="awaiting migration T2a3", strict=False)
def test_normal_vector_z_quarter_pi(golden):
    """Rotating the z-axis around z by pi/4 → still [0, 0, 1]."""
    from imatools.core.spatial import create_normal_vector_for_plane

    result = create_normal_vector_for_plane("z", np.pi / 4)
    expected = golden("spatial/normal_vector_z_quarter_pi")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


@pytest.mark.xfail(reason="awaiting migration T2a3", strict=False)
def test_normal_vector_x_half_pi(golden):
    """Rotating the x-axis around x by pi/2 → still [1, 0, 0]."""
    from imatools.core.spatial import create_normal_vector_for_plane

    result = create_normal_vector_for_plane("x", np.pi / 2)
    expected = golden("spatial/normal_vector_x_half_pi")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


@pytest.mark.xfail(reason="awaiting migration T2a3", strict=False)
def test_normal_vector_y_pi(golden):
    """Rotating the y-axis around y by pi → still [0, 1, 0]."""
    from imatools.core.spatial import create_normal_vector_for_plane

    result = create_normal_vector_for_plane("y", np.pi)
    expected = golden("spatial/normal_vector_y_pi")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# create_image_at_plane — BROKEN IN MASTER (see module docstring)
# ---------------------------------------------------------------------------
#
# master's ``create_image_at_plane_from_vector`` always raises:
#     sitk::ERROR: Length of input (3) does not match matrix dimensions (3, 3)
# because ``transform.SetMatrix(normal_vector + [0, 0, 1])`` passes 3 elements
# instead of 9.  No golden file exists; this test is a stub for T2a3.


@pytest.mark.xfail(reason="awaiting migration T2a3", strict=False)
def test_create_image_at_plane_import():
    """Target module can be imported and ``create_image_at_plane`` is accessible."""
    from imatools.core.spatial import create_image_at_plane  # noqa: F401


@pytest.mark.xfail(reason="awaiting migration T2a3", strict=False)
def test_create_image_at_plane_returns_2d_array():
    """``create_image_at_plane`` returns a 2-D numpy array for valid inputs.

    NOTE for T2a3: fix ``transform.SetMatrix(normal_vector + [0, 0, 1])`` —
    concatenate a 9-element sequence (e.g. using ``np.concatenate`` or a proper
    rotation-matrix construction) instead of adding [0, 0, 1] elementwise.
    """
    from imatools.core.spatial import create_image_at_plane

    img = _make_label_image()
    point = np.array([6.0, 6.0, 6.0])
    result = create_image_at_plane(img, point, "z", 0.0)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


# ---------------------------------------------------------------------------
# create_image_at_plane_from_vector — BROKEN IN MASTER (see module docstring)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a3", strict=False)
def test_create_image_at_plane_from_vector_import():
    """Target module can be imported and ``create_image_at_plane_from_vector`` is accessible."""
    from imatools.core.spatial import create_image_at_plane_from_vector  # noqa: F401


@pytest.mark.xfail(reason="awaiting migration T2a3", strict=False)
def test_create_image_at_plane_from_vector_returns_2d_array():
    """``create_image_at_plane_from_vector`` returns a 2-D numpy array.

    NOTE for T2a3: see ``test_create_image_at_plane_returns_2d_array`` for the fix.
    """
    from imatools.core.spatial import create_image_at_plane_from_vector

    img = _make_label_image()
    point = np.array([6.0, 6.0, 6.0])
    normal_vector = np.array([0.0, 0.0, 1.0])
    result = create_image_at_plane_from_vector(img, point, normal_vector)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


# ---------------------------------------------------------------------------
# set_direction_as
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a3", strict=False)
def test_set_direction_as_basic(golden):
    """Direction is copied from ref; spacing and origin remain from im."""
    from imatools.core.spatial import set_direction_as

    im = _make_label_image()
    ref = _make_ref_image()
    result = set_direction_as(im, ref)
    expected = golden("spatial/set_direction_as_basic")

    assert list(result.GetDirection()) == pytest.approx(expected["direction"], rel=1e-7)
    assert list(result.GetSpacing()) == pytest.approx(expected["spacing"], rel=1e-7)
    assert list(result.GetOrigin()) == pytest.approx(expected["origin"], rel=1e-7)


# ---------------------------------------------------------------------------
# fix_header_to_axis_aligned
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a3", strict=False)
def test_fix_header_oblique(golden):
    """Off-diagonal space directions are replaced by a diagonal of their row norms."""
    from imatools.core.spatial import fix_header_to_axis_aligned

    result = fix_header_to_axis_aligned(_HDR_OBLIQUE)
    expected = golden("spatial/fix_header_oblique")

    np.testing.assert_allclose(
        np.asarray(result["space directions"]),
        np.asarray(expected["space directions"]),
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result["space origin"]),
        np.asarray(expected["space origin"]),
        rtol=1e-7,
    )
    assert result["srow_x"] == expected["srow_x"]
    assert result["srow_y"] == expected["srow_y"]
    assert result["srow_z"] == expected["srow_z"]


@pytest.mark.xfail(reason="awaiting migration T2a3", strict=False)
def test_fix_header_aligned(golden):
    """Already axis-aligned header passes through unchanged (identity case)."""
    from imatools.core.spatial import fix_header_to_axis_aligned

    result = fix_header_to_axis_aligned(_HDR_ALIGNED)
    expected = golden("spatial/fix_header_aligned")

    np.testing.assert_allclose(
        np.asarray(result["space directions"]),
        np.asarray(expected["space directions"]),
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result["space origin"]),
        np.asarray(expected["space origin"]),
        rtol=1e-7,
    )
    assert result["srow_x"] == expected["srow_x"]
    assert result["srow_y"] == expected["srow_y"]
    assert result["srow_z"] == expected["srow_z"]
