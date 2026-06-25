"""Characterization tests for ``imatools.core.geometry`` (T1d).

All tests import from the TARGET location ``imatools.core.geometry``.  That module
does not exist yet — it will be created by migration task T2b1.  Until then every
test is marked ``xfail(strict=False)`` so it is collected but does not block CI.

Functions characterized here:

  From master ``imatools/common/vtktools.py``:
    ``l2_norm`` (the vtktools variant — per-row ``np.linalg.norm(..., axis=1)``),
    ``dot_prod_vec``, ``point_in_aabb``, ``point_in_aabb_vectorized``,
    ``get_bounding_box``, ``precompute_valid_cells``, ``get_cog_per_element``,
    ``compute_mesh_size``.

  From master ``imatools/common/utils.py``:
    ``rotation_matrix``.

Golden values were captured from master via::

    M=~/dev/python/imatools.worktrees/master
    ~/opt/anaconda3/bin/conda run -n imatools env PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module geometry --out tests/golden

Comparison helpers
------------------
* **npy** goldens  -> numpy arrays; compared with ``np.testing.assert_allclose``.
* **json** goldens -> Python scalars/lists; booleans compared with ``==``.
"""

from __future__ import annotations

import math

import _fixtures as fx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixture-derived constants (mirror geometry.py capture constants)
# ---------------------------------------------------------------------------

_VF = fx.vector_field()  # shape (16, 3)
_VF_REV = _VF[::-1].copy()

_AXIS = np.array([0.0, 0.0, 1.0])
_THETA = math.pi / 4

_PD = fx.polydata()

_POINT_INSIDE = np.array([0.5, 0.5, 0.5])
_POINT_OUTSIDE = np.array([2.0, 2.0, 2.0])
_BOX_CORNERS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
)

_POINTS = np.array(
    [
        [0.5, 0.5, 0.5],
        [2.0, 2.0, 2.0],
        [0.1, 0.1, 0.1],
        [1.5, 1.5, 1.5],
    ],
    dtype=float,
)
_BOXES_MINMAX = [
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
]


# ---------------------------------------------------------------------------
# l2_norm (vtktools variant — axis=1)
# ---------------------------------------------------------------------------


def test_l2_norm_rows(golden):
    from imatools.core.geometry import l2_norm

    result = l2_norm(_VF)
    expected = golden("geometry/l2_norm_rows")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_l2_norm_single_row(golden):
    from imatools.core.geometry import l2_norm

    result = l2_norm(np.array([[3.0, 4.0, 0.0]]))
    expected = golden("geometry/l2_norm_single_row")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# dot_prod_vec
# ---------------------------------------------------------------------------


def test_dot_prod_vec_basic(golden):
    from imatools.core.geometry import dot_prod_vec

    result = dot_prod_vec(_VF, _VF_REV)
    expected = golden("geometry/dot_prod_vec_basic")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_dot_prod_vec_self(golden):
    from imatools.core.geometry import dot_prod_vec

    result = dot_prod_vec(_VF, _VF)
    expected = golden("geometry/dot_prod_vec_self")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# point_in_aabb
# ---------------------------------------------------------------------------


def test_point_in_aabb_inside(golden):
    from imatools.core.geometry import point_in_aabb

    result = point_in_aabb(_POINT_INSIDE, _BOX_CORNERS)
    expected = golden("geometry/point_in_aabb_inside")
    assert bool(result) == bool(expected)


def test_point_in_aabb_outside(golden):
    from imatools.core.geometry import point_in_aabb

    result = point_in_aabb(_POINT_OUTSIDE, _BOX_CORNERS)
    expected = golden("geometry/point_in_aabb_outside")
    assert bool(result) == bool(expected)


def test_point_in_aabb_boundary(golden):
    from imatools.core.geometry import point_in_aabb

    result = point_in_aabb(np.array([0.0, 0.0, 0.0]), _BOX_CORNERS)
    expected = golden("geometry/point_in_aabb_boundary")
    assert bool(result) == bool(expected)


# ---------------------------------------------------------------------------
# point_in_aabb_vectorized
# ---------------------------------------------------------------------------


def test_point_in_aabb_vectorized_mixed(golden):
    from imatools.core.geometry import point_in_aabb_vectorized

    result = point_in_aabb_vectorized(_POINTS, _BOXES_MINMAX)
    expected = golden("geometry/point_in_aabb_vectorized_mixed")
    np.testing.assert_array_equal(result.astype(np.uint8), expected)


def test_point_in_aabb_vectorized_empty_boxes(golden):
    from imatools.core.geometry import point_in_aabb_vectorized

    result = point_in_aabb_vectorized(_POINTS, [])
    expected = golden("geometry/point_in_aabb_vectorized_empty_boxes")
    np.testing.assert_array_equal(result.astype(np.uint8), expected)


# ---------------------------------------------------------------------------
# get_bounding_box
# ---------------------------------------------------------------------------


def test_get_bounding_box_polydata(golden):
    from imatools.core.geometry import get_bounding_box

    result = get_bounding_box(_PD)
    expected = golden("geometry/get_bounding_box_polydata")
    assert list(result) == pytest.approx(expected, rel=1e-7)


# ---------------------------------------------------------------------------
# get_cog_per_element
# ---------------------------------------------------------------------------


def test_get_cog_per_element_polydata(golden):
    from imatools.core.geometry import get_cog_per_element

    result = get_cog_per_element(_PD)
    expected = golden("geometry/get_cog_per_element_polydata")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# precompute_valid_cells
# ---------------------------------------------------------------------------


def test_precompute_valid_cells_basic(golden):
    from imatools.core.geometry import get_cog_per_element, precompute_valid_cells

    cogs = get_cog_per_element(_PD)
    result = precompute_valid_cells(cogs, _BOXES_MINMAX)
    expected = golden("geometry/precompute_valid_cells_basic")
    assert sorted(result) == expected


def test_precompute_valid_cells_empty(golden):
    from imatools.core.geometry import get_cog_per_element, precompute_valid_cells

    cogs = get_cog_per_element(_PD)
    result = precompute_valid_cells(cogs, [])
    expected = golden("geometry/precompute_valid_cells_empty")
    assert sorted(result) == expected


# ---------------------------------------------------------------------------
# compute_mesh_size
# ---------------------------------------------------------------------------


def test_compute_mesh_size_polydata(golden):
    from imatools.core.geometry import compute_mesh_size

    result = compute_mesh_size(_PD)
    expected = golden("geometry/compute_mesh_size_polydata")
    assert list(result) == pytest.approx(expected, rel=1e-7)


# ---------------------------------------------------------------------------
# rotation_matrix
# ---------------------------------------------------------------------------


def test_rotation_matrix_z45(golden):
    from imatools.core.geometry import rotation_matrix

    result = rotation_matrix(_AXIS, _THETA)
    expected = golden("geometry/rotation_matrix_z45")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_rotation_matrix_zero(golden):
    from imatools.core.geometry import rotation_matrix

    result = rotation_matrix(np.array([1.0, 0.0, 0.0]), 0.0)
    expected = golden("geometry/rotation_matrix_zero")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_rotation_matrix_x180(golden):
    from imatools.core.geometry import rotation_matrix

    result = rotation_matrix(np.array([1.0, 0.0, 0.0]), math.pi)
    expected = golden("geometry/rotation_matrix_x180")
    np.testing.assert_allclose(result, expected, rtol=1e-7)
