"""Capture cases for ``imatools.core.geometry`` (T1d).

Functions sourced from:
- master ``imatools/common/vtktools.py``:
    ``l2_norm``, ``dot_prod_vec``, ``point_in_aabb``, ``point_in_aabb_vectorized``,
    ``get_bounding_box``, ``precompute_valid_cells``, ``get_cog_per_element``,
    ``compute_mesh_size``
- master ``imatools/common/utils.py``:
    ``rotation_matrix``

NOTE on ``l2_norm`` disambiguation: ``ioutils.l2_norm`` (already in the ``metrics``
group, T1f) applies ``np.linalg.norm`` with no ``axis`` keyword.  This one comes from
``vtktools`` and uses ``axis=1``, producing per-row norms of a 2-D array.

NOTE on ``get_cog_per_element`` + ``extractPointsAndElemsFromVtk``: the helper
hard-codes ``range(3)`` for point IDs, so it is valid ONLY for triangular meshes
(3 points per cell).  We feed ``fx.polydata()`` (all triangles) — not
``fx.unstructured_grid()`` (tetrahedra with 4 points).
"""

from __future__ import annotations

import math

import _fixtures as fx
import numpy as np
from _capture_golden import CaptureCase

from imatools.common import utils as common_utils
from imatools.common import vtktools

# ---------------------------------------------------------------------------
# Shared fixture-derived constants
# ---------------------------------------------------------------------------

# A (N, 3) array of 2-D row vectors for l2_norm / dot_prod_vec tests.
_VF = fx.vector_field()  # shape (16, 3)

# A second vector field (reversed) for dot_prod_vec.
_VF_REV = _VF[::-1].copy()

# Rotation axis (unit vector along z) and angle.
_AXIS = np.array([0.0, 0.0, 1.0])
_THETA = math.pi / 4  # 45 degrees

# Pre-built polydata mesh (triangulated surface) — safe for get_cog_per_element.
_PD = fx.polydata()

# A single point and a box (8 corners) for point_in_aabb.
_POINT_INSIDE = np.array([0.5, 0.5, 0.5])
_POINT_OUTSIDE = np.array([2.0, 2.0, 2.0])
# 8 corners of a unit cube [0,1]^3
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

# Points array and bounding boxes (in min/max form) for vectorized / precompute tests.
_POINTS = np.array(
    [
        [0.5, 0.5, 0.5],  # inside box 0
        [2.0, 2.0, 2.0],  # outside all boxes
        [0.1, 0.1, 0.1],  # inside box 0
        [1.5, 1.5, 1.5],  # inside box 1
    ],
    dtype=float,
)
# Two bounding boxes in (min_x, min_y, min_z, max_x, max_y, max_z) form.
_BOXES_MINMAX = [
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
]

# ---------------------------------------------------------------------------
# Reducers for VTK / structured outputs
# ---------------------------------------------------------------------------


def _cog_reduce(cogs: np.ndarray) -> np.ndarray:
    """COGs are already an ndarray; pass through for npy serialization."""
    return cogs


def _bounding_box_reduce(result):
    """Return bounding box tuple as a list for JSON serialization."""
    return list(result)


def _compute_mesh_size_reduce(result):
    """Return (num_cells, total_area) tuple as a list for JSON serialization."""
    return list(result)


def _precompute_reduce(result) -> list:
    """Convert the set of valid cell indices to a sorted list for JSON."""
    return sorted(result)


CASES = [
    # ------------------------------------------------------------------
    # l2_norm (vtktools version — per-row norms, axis=1)
    # ------------------------------------------------------------------
    CaptureCase(
        name="geometry/l2_norm_rows",
        func=vtktools.l2_norm,
        args=(_VF,),
        fmt="npy",
    ),
    # Edge case: single row
    CaptureCase(
        name="geometry/l2_norm_single_row",
        func=vtktools.l2_norm,
        args=(np.array([[3.0, 4.0, 0.0]]),),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # dot_prod_vec — element-wise dot product of two (N,3) arrays
    # ------------------------------------------------------------------
    CaptureCase(
        name="geometry/dot_prod_vec_basic",
        func=vtktools.dot_prod_vec,
        args=(_VF, _VF_REV),
        fmt="npy",
    ),
    # Edge case: identical arrays -> squared norms
    CaptureCase(
        name="geometry/dot_prod_vec_self",
        func=vtktools.dot_prod_vec,
        args=(_VF, _VF),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # point_in_aabb — scalar bool, serialized as JSON
    # ------------------------------------------------------------------
    CaptureCase(
        name="geometry/point_in_aabb_inside",
        func=vtktools.point_in_aabb,
        args=(_POINT_INSIDE, _BOX_CORNERS),
        reduce=bool,
        fmt="json",
    ),
    CaptureCase(
        name="geometry/point_in_aabb_outside",
        func=vtktools.point_in_aabb,
        args=(_POINT_OUTSIDE, _BOX_CORNERS),
        reduce=bool,
        fmt="json",
    ),
    # Edge case: point on the boundary (corner itself)
    CaptureCase(
        name="geometry/point_in_aabb_boundary",
        func=vtktools.point_in_aabb,
        args=(np.array([0.0, 0.0, 0.0]), _BOX_CORNERS),
        reduce=bool,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # point_in_aabb_vectorized — boolean ndarray output
    # ------------------------------------------------------------------
    CaptureCase(
        name="geometry/point_in_aabb_vectorized_mixed",
        func=vtktools.point_in_aabb_vectorized,
        args=(_POINTS, _BOXES_MINMAX),
        reduce=lambda arr: arr.astype(np.uint8),
        fmt="npy",
    ),
    # Edge case: empty box list -> all False
    CaptureCase(
        name="geometry/point_in_aabb_vectorized_empty_boxes",
        func=vtktools.point_in_aabb_vectorized,
        args=(_POINTS, []),
        reduce=lambda arr: arr.astype(np.uint8),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # get_bounding_box — returns (min_x, min_y, min_z, max_x, max_y, max_z)
    # ------------------------------------------------------------------
    CaptureCase(
        name="geometry/get_bounding_box_polydata",
        func=vtktools.get_bounding_box,
        args=(_PD,),
        reduce=_bounding_box_reduce,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # get_cog_per_element — ndarray of centroids
    # Only safe on triangulated mesh (extractPointsAndElemsFromVtk uses range(3))
    # ------------------------------------------------------------------
    CaptureCase(
        name="geometry/get_cog_per_element_polydata",
        func=vtktools.get_cog_per_element,
        args=(_PD,),
        reduce=_cog_reduce,
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # precompute_valid_cells — returns a Set[int]
    # Feed COGs derived from the polydata and the same BOXES_MINMAX above.
    # ------------------------------------------------------------------
    CaptureCase(
        name="geometry/precompute_valid_cells_basic",
        func=vtktools.precompute_valid_cells,
        args=(
            vtktools.get_cog_per_element(_PD),
            _BOXES_MINMAX,
        ),
        reduce=_precompute_reduce,
        fmt="json",
    ),
    # Edge case: no bounding boxes -> empty set
    CaptureCase(
        name="geometry/precompute_valid_cells_empty",
        func=vtktools.precompute_valid_cells,
        args=(
            vtktools.get_cog_per_element(_PD),
            [],
        ),
        reduce=_precompute_reduce,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # compute_mesh_size — returns (num_cells, total_area)
    # ------------------------------------------------------------------
    CaptureCase(
        name="geometry/compute_mesh_size_polydata",
        func=vtktools.compute_mesh_size,
        args=(_PD,),
        reduce=_compute_mesh_size_reduce,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # rotation_matrix — 3x3 rotation matrix
    # ------------------------------------------------------------------
    CaptureCase(
        name="geometry/rotation_matrix_z45",
        func=common_utils.rotation_matrix,
        args=(_AXIS, _THETA),
        fmt="npy",
    ),
    # Edge case: zero rotation -> identity matrix
    CaptureCase(
        name="geometry/rotation_matrix_zero",
        func=common_utils.rotation_matrix,
        args=(np.array([1.0, 0.0, 0.0]), 0.0),
        fmt="npy",
    ),
    # Edge case: 180 degrees around x-axis
    CaptureCase(
        name="geometry/rotation_matrix_x180",
        func=common_utils.rotation_matrix,
        args=(np.array([1.0, 0.0, 0.0]), math.pi),
        fmt="npy",
    ),
]
