# src/imatools/core/geometry.py
"""Pure geometry functions migrated from ``imatools.common.vtktools`` and
``imatools.common.utils`` (T2b1).

The 9 public functions here are the authoritative implementations; the old
``imatools.common.vtktools`` and ``imatools.common.utils`` modules re-export
them via shims at their bottoms.

Functions that remain in ``vtktools`` (``extractPointsAndElemsFromVtk``,
``mesh_to_image``) are accessed lazily at call time via ``_vtk()`` to avoid
circular imports: vtktools must finish defining its own helpers before its
bottom shim imports this module; therefore this module must NOT import from
``vtktools`` at module-load time.

Bug notes
---------
- ``get_cog_per_element``: relies on ``extractPointsAndElemsFromVtk`` which
  hard-codes ``range(3)`` — truncates non-triangle/tet cells to 3 points.
  Cat-B latent bug, preserved VERBATIM; golden locks this behaviour.
"""

from __future__ import annotations

import math
from typing import List, Set

import numpy as np

from imatools.common.config import configure_logging

logger = configure_logging(log_name=__name__)


# ---------------------------------------------------------------------------
# Lazy-helper accessor — avoids circular import at module load time.
# After vtktools finishes loading (including its bottom shim), all helper
# names are available in sys.modules and these lookups resolve instantly.
# ---------------------------------------------------------------------------
def _vtk():
    """Return the vtktools module (always already loaded when a geometry fn is called)."""
    import imatools.common.vtktools as _m  # noqa: PLC0415

    return _m


# ---------------------------------------------------------------------------
# Geometry functions — verbatim from master imatools/common/vtktools.py
# ---------------------------------------------------------------------------


def l2_norm(a):
    return np.linalg.norm(a, axis=1)


def dot_prod_vec(a, b):
    return np.sum(a * b, axis=1)


def get_cog_per_element(msh) -> np.ndarray:
    pts, el = _vtk().extractPointsAndElemsFromVtk(msh)
    element_coordinates = pts[el]

    cog = np.mean(element_coordinates, axis=1)

    return cog


def get_bounding_box(msh):
    """
    Get the bounding box of a mesh.
    Returns a tuple of (min_x, min_y, min_z, max_x, max_y, max_z).
    """
    bounds = msh.GetBounds()
    return (bounds[0], bounds[2], bounds[4], bounds[1], bounds[3], bounds[5])


def point_in_aabb(point, box_corners):
    """
    Check if a point lies within the axis-aligned bounding box defined by the 8 voxel corners.
    """
    mins = np.min(box_corners, axis=0)
    maxs = np.max(box_corners, axis=0)
    return np.all(point >= mins) and np.all(point <= maxs)


def point_in_aabb_vectorized(points: np.ndarray, boxes: List) -> np.ndarray:
    """
    Vectorized version to check if points are in any bounding box.

    Args:
        points: Nx3 array of points
        boxes: List of bounding boxes

    Returns:
        Boolean array indicating which points are in any box
    """
    if len(boxes) == 0:
        return np.zeros(len(points), dtype=bool)

    # Convert boxes to numpy array for vectorized operations
    # Assuming boxes are in format [(min_x, min_y, min_z, max_x, max_y, max_z), ...]
    boxes_array = np.array(boxes)
    points_in_any_box = np.zeros(len(points), dtype=bool)

    for box in boxes_array:
        if box.shape == (8, 3):  # 8 corners
            min_coords = box.min(axis=0)
            max_coords = box.max(axis=0)
        elif box.shape == (6,):  # already in min/max form
            min_coords, max_coords = box[:3], box[3:]
        elif box.shape == (2, 3):  # explicit [min,max]
            min_coords, max_coords = box[0], box[1]
        else:
            raise ValueError(f"Unexpected box shape: {box.shape}")

        # Vectorized check for all points in this box
        in_box = np.all((points >= min_coords) & (points <= max_coords), axis=1)
        points_in_any_box |= in_box

    return points_in_any_box


def precompute_valid_cells(cogs: np.ndarray, voxel_bounding_boxes: List) -> Set[int]:
    """
    Pre-identify which cells are within bounding boxes.

    Args:
        cogs: Nx3 array of cell centers of gravity
        voxel_bounding_boxes: List of bounding boxes

    Returns:
        Set of valid cell indices
    """
    valid_mask = point_in_aabb_vectorized(cogs, voxel_bounding_boxes)
    return set(np.where(valid_mask)[0])


def compute_mesh_size(msh) -> tuple:
    """
    Compute the size of a mesh by calculating the sum of the areas of all cells.
    """
    total_area = 0.0
    for i in range(msh.GetNumberOfCells()):
        cell = msh.GetCell(i)
        total_area += cell.ComputeArea()

    return msh.GetNumberOfCells(), total_area


# ---------------------------------------------------------------------------
# rotation_matrix — verbatim from master imatools/common/utils.py
# ---------------------------------------------------------------------------


def rotation_matrix(u: np.ndarray, theta: float) -> np.ndarray:
    """
    Calculate the rotation matrix for a given axis and angle.

    Parameters:
    - u (array-like): 3-element array representing the rotation axis.
    - theta (float): Angle of rotation in radians.

    Returns:
    - array: 3x3 rotation matrix.
    """
    R = np.zeros((3, 3), dtype=float)  # noqa: N806 — verbatim from master utils.py
    R[0, 0] = u[0] ** 2 + math.cos(theta) * (1 - u[0] ** 2)
    R[0, 1] = (1 - math.cos(theta)) * u[0] * u[1] - u[2] * math.sin(theta)
    R[0, 2] = (1 - math.cos(theta)) * u[0] * u[2] + u[1] * math.sin(theta)

    R[1, 0] = (1 - math.cos(theta)) * u[0] * u[1] + u[2] * math.sin(theta)
    R[1, 1] = u[1] ** 2 + math.cos(theta) * (1 - u[1] ** 2)
    R[1, 2] = (1 - math.cos(theta)) * u[1] * u[2] - u[0] * math.sin(theta)

    R[2, 0] = (1 - math.cos(theta)) * u[0] * u[2] - u[1] * math.sin(theta)
    R[2, 1] = (1 - math.cos(theta)) * u[1] * u[2] + u[0] * math.sin(theta)
    R[2, 2] = u[2] ** 2 + math.cos(theta) * (1 - u[2] ** 2)

    return R
