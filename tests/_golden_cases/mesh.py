"""Capture cases for ``imatools.core.mesh`` (T1e).

Functions sourced from:
- master ``imatools/common/vtktools.py`` (mesh transforms set):
    ``clean_mesh``, ``genericThreshold``, ``thresholdExactValue``,
    ``set_cell_to_point_data``, ``setCellDataToPointData``,
    ``point_to_cell_data``, ``cell_to_point_data``,
    ``set_vtk_scalars``, ``set_cell_scalars``, ``indices_at_scalar``,
    ``convertPointDataToNpArray``, ``convertCellDataToNpArray``,
    ``getSurfaceArea``, ``getElemPermutation``,
    ``fibrosis_score``, ``fibrosis_score_cell``, ``fibrosis_score_point``,
    ``fibrorisScore`` (legacy name),
    ``fibrosis_overlap``, ``fibrosis_overlap_points``, ``fibrosis_overlap_cells``,
    ``fibrosisOverlapCell``,
    ``tag_elements_by_voxel_boxes``, ``tag_mesh_elements_by_voxel_boxes``,
    ``tag_mesh_elements_by_growing_from_seed``,
    ``tag_mesh_elements_by_growing_from_seed_optimized`` (uses master's NEW signature
    with ``label_value`` param — RECONCILIATION: master is the oracle),
    ``tag_mesh_elements_parallel_regions``,
    ``flip_xy``, ``global_centre_of_mass``, ``translate_to_point``, ``join_vtk``,
    ``map_cells``, ``map_points``, ``compare_mesh_sizes``,
    ``ugrid2polydata``, ``cogs_from_ugrid``, ``extractPointsAndElemsFromVtk``,
    ``get_element_cogs``,
    ``build_adjacency_list``, ``build_adjacency_list_optimized``,
    ``np_to_vtk_array``, ``mask_cell_scalars``,
    ``verify_cell_indices``, ``verify_cell_indices_from_mesh``.
- master ``imatools/common/utils.py``:
    ``rotate_mesh``

NOTES:
- ``compare_mesh_sizes`` reads VTK files from disk; we write temp files inline.
- ``rotate_mesh`` requires a pyvista mesh with cells encoded as
  [4, id0, id1, id2, id3] and ``cell_data["ID"]`` tags; built inline.
- ``getElemPermutation`` is a stub in master (always returns 0 or -1); characterized
  as-is.
- ``fibrosis_overlap*`` functions require meshes that share the SAME number of
  cells/points and have scalar arrays; built inline with matching polydata pairs.
- ``tag_mesh_elements_by_growing_from_seed*`` need a VTK mesh, seed points (Nx3),
  and voxel bounding boxes; built inline using the unstructured_grid fixture with
  seeds at known cell centroids.
"""

from __future__ import annotations

import os
import tempfile

import _fixtures as fx
import numpy as np
import vtk
import vtk.util.numpy_support as vtknp
from _capture_golden import CaptureCase

from imatools.common import utils as common_utils
from imatools.common import vtktools

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PD = fx.polydata()  # 5 points, 4 triangles, has scalars
_UG = fx.unstructured_grid()  # 5 points, 2 tetras, has scalars


# ---------------------------------------------------------------------------
# Helper: polydata pair with matching cell counts and scalar arrays
# Used for fibrosis_overlap* and fibrosis_score* functions
# ---------------------------------------------------------------------------


def _make_scored_polydata():
    """Return a triangulated polydata with a float cell scalar ramp [0..1]."""
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(0.0, 0.0, 0.0)
    pts.InsertNextPoint(1.0, 0.0, 0.0)
    pts.InsertNextPoint(0.0, 1.0, 0.0)
    pts.InsertNextPoint(1.0, 1.0, 0.0)
    pts.InsertNextPoint(0.5, 0.5, 1.0)
    pd.SetPoints(pts)

    cells = vtk.vtkCellArray()
    for a, b, c in [(0, 1, 2), (1, 3, 2), (0, 2, 4), (1, 0, 4)]:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, a)
        tri.GetPointIds().SetId(1, b)
        tri.GetPointIds().SetId(2, c)
        cells.InsertNextCell(tri)
    pd.SetPolys(cells)
    return pd


def _attach_cell_scalars(pd, values):
    """Attach a float scalar array (named 'scalars') to pd's cell data."""
    arr = vtk.vtkFloatArray()
    arr.SetName("scalars")
    for v in values:
        arr.InsertNextValue(v)
    pd.GetCellData().SetScalars(arr)
    return pd


# msh0 and msh1 share the same topology; scalars are [0.3, 0.6, 0.9, 0.4] and
# [0.2, 0.7, 0.5, 0.8] respectively. Threshold 0.5 used in fibrosis tests.
_FIBROSIS_MSH0 = _attach_cell_scalars(_make_scored_polydata(), [0.3, 0.6, 0.9, 0.4])
_FIBROSIS_MSH1 = _attach_cell_scalars(_make_scored_polydata(), [0.2, 0.7, 0.5, 0.8])
_FIBROSIS_TH = 0.5


# ---------------------------------------------------------------------------
# Helper: bounding boxes and seeds for the tagging functions
# The unstructured grid has 2 tetra cells. We use bounding boxes that
# enclose each tetra's approximate centroid.
# ---------------------------------------------------------------------------


def _ug_cog(ug, cell_id):
    """Return the centroid of a cell in the unstructured grid."""
    cell = ug.GetCell(cell_id)
    n = cell.GetNumberOfPoints()
    cog = np.zeros(3)
    for i in range(n):
        cog += np.array(ug.GetPoint(cell.GetPointId(i)))
    return cog / n


_UG_COG0 = _ug_cog(_UG, 0)  # centroid of first tetra
_UG_COG1 = _ug_cog(_UG, 1)  # centroid of second tetra

# Bounding boxes as 8-corner arrays (8x3) that contain respective COGs.
_EPS = 0.3


def _box_around(cog, eps=_EPS):
    c = np.array(cog)
    corners = np.array(
        [
            [c[0] - eps, c[1] - eps, c[2] - eps],
            [c[0] + eps, c[1] - eps, c[2] - eps],
            [c[0] - eps, c[1] + eps, c[2] - eps],
            [c[0] + eps, c[1] + eps, c[2] - eps],
            [c[0] - eps, c[1] - eps, c[2] + eps],
            [c[0] + eps, c[1] - eps, c[2] + eps],
            [c[0] - eps, c[1] + eps, c[2] + eps],
            [c[0] + eps, c[1] + eps, c[2] + eps],
        ]
    )
    return corners


_SEED_POINTS = np.array([_UG_COG0, _UG_COG1])
_VOXEL_BOXES = [_box_around(_UG_COG0), _box_around(_UG_COG1)]


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


def _pd_reduce(pd):
    """Reduce polydata to (n_pts, n_cells, scalar_min, scalar_max)."""
    n_pts = pd.GetNumberOfPoints()
    n_cells = pd.GetNumberOfCells()
    scalars = pd.GetCellData().GetScalars()
    if scalars is not None:
        arr = vtknp.vtk_to_numpy(scalars)
        return {
            "n_points": int(n_pts),
            "n_cells": int(n_cells),
            "scalar_min": float(arr.min()),
            "scalar_max": float(arr.max()),
        }
    return {"n_points": int(n_pts), "n_cells": int(n_cells)}


def _ug_reduce(ug):
    """Reduce unstructured grid to (n_pts, n_cells, scalar_min, scalar_max)."""
    n_pts = ug.GetNumberOfPoints()
    n_cells = ug.GetNumberOfCells()
    scalars = ug.GetCellData().GetScalars()
    if scalars is not None:
        arr = vtknp.vtk_to_numpy(scalars)
        return {
            "n_points": int(n_pts),
            "n_cells": int(n_cells),
            "scalar_min": float(arr.min()),
            "scalar_max": float(arr.max()),
        }
    return {"n_points": int(n_pts), "n_cells": int(n_cells)}


def _pd_reduce_with_named_scalars(name):
    """Return a reducer that reads a named cell array from the polydata."""

    def _reduce(pd):
        n_pts = pd.GetNumberOfPoints()
        n_cells = pd.GetNumberOfCells()
        arr_vtk = pd.GetCellData().GetArray(name)
        if arr_vtk is not None:
            arr = vtknp.vtk_to_numpy(arr_vtk)
            return {
                "n_points": int(n_pts),
                "n_cells": int(n_cells),
                "field_name": name,
                "values": arr.tolist(),
            }
        return {"n_points": int(n_pts), "n_cells": int(n_cells), "field_name": name}

    return _reduce


def _ug_scar_reduce(name="scar"):
    """Reducer for tagged UG: return cell tag array."""

    def _reduce(ug):
        arr_vtk = ug.GetCellData().GetArray(name)
        if arr_vtk is not None:
            arr = vtknp.vtk_to_numpy(arr_vtk)
            return {"tags": arr.tolist(), "n_cells": ug.GetNumberOfCells()}
        return {"tags": [], "n_cells": ug.GetNumberOfCells()}

    return _reduce


def _fibrosis_count_reduce(result):
    """Reduce (mesh, count_dict) returned by fibrosis_overlap*."""
    _msh, count_dic = result
    return {k: int(v) if isinstance(v, (int, np.integer)) else v for k, v in count_dic.items()}


def _map_reduce(dic):
    """Reduce mapping dict: keep only integer arrays (ids) and distances."""
    out = {}
    for k, v in dic.items():
        arr = np.asarray(v)
        if arr.dtype.kind in ("i", "u"):
            out[k] = arr.tolist()
        else:
            out[k] = float(arr.mean())  # reduce float columns to mean
    return out


def _compare_mesh_sizes_reduce(result):
    """Reduce (path_large, path_small, tot_large, tot_small, large_id, small_id)."""
    if result[0] is None:
        return None
    return {
        "tot_large": int(result[2]),
        "tot_small": int(result[3]),
        "large_id": result[4],
        "small_id": result[5],
    }


# ---------------------------------------------------------------------------
# compare_mesh_sizes: needs VTK files on disk — write and clean up inline
# ---------------------------------------------------------------------------


def _compare_mesh_sizes_pts():
    """Write two VTK polydata files and call compare_mesh_sizes(pts mode)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = os.path.join(tmpdir, "a.vtk")
        path_b = os.path.join(tmpdir, "b.vtk")
        vtktools.write_vtk(_PD, tmpdir, "a", output_type="polydata")
        # Build a second polydata with fewer points
        pd2 = vtk.vtkPolyData()
        pts2 = vtk.vtkPoints()
        pts2.InsertNextPoint(0.0, 0.0, 0.0)
        pts2.InsertNextPoint(1.0, 0.0, 0.0)
        pts2.InsertNextPoint(0.0, 1.0, 0.0)
        pd2.SetPoints(pts2)
        cells2 = vtk.vtkCellArray()
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, 0)
        tri.GetPointIds().SetId(1, 1)
        tri.GetPointIds().SetId(2, 2)
        cells2.InsertNextCell(tri)
        pd2.SetPolys(cells2)
        vtktools.write_vtk(pd2, tmpdir, "b", output_type="polydata")
        result = vtktools.compare_mesh_sizes(path_a, path_b, "A", "B", 0)
    return result


def _compare_mesh_sizes_elem():
    """Write two VTK polydata files and call compare_mesh_sizes(elem mode)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = os.path.join(tmpdir, "a.vtk")
        path_b = os.path.join(tmpdir, "b.vtk")
        vtktools.write_vtk(_PD, tmpdir, "a", output_type="polydata")
        pd2 = vtk.vtkPolyData()
        pts2 = vtk.vtkPoints()
        pts2.InsertNextPoint(0.0, 0.0, 0.0)
        pts2.InsertNextPoint(1.0, 0.0, 0.0)
        pts2.InsertNextPoint(0.0, 1.0, 0.0)
        pd2.SetPoints(pts2)
        cells2 = vtk.vtkCellArray()
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, 0)
        tri.GetPointIds().SetId(1, 1)
        tri.GetPointIds().SetId(2, 2)
        cells2.InsertNextCell(tri)
        pd2.SetPolys(cells2)
        vtktools.write_vtk(pd2, tmpdir, "b", output_type="polydata")
        result = vtktools.compare_mesh_sizes(path_a, path_b, "A", "B", 1)
    return result


# ---------------------------------------------------------------------------
# rotate_mesh: requires pyvista mesh with specific structure
# We build a minimal pyvista UnstructuredGrid with 3 cells labelled 1, 7, 8
# so lv_tag=1, mv_tag=7, tv_tag=8 are all present.
# The function modifies plt_msh.points in place and returns plt_msh.
# ---------------------------------------------------------------------------


def _make_rotate_mesh_input():
    """Build a minimal pyvista mesh suitable for rotate_mesh."""
    import pyvista as pv

    # 8 points forming a simple shape with identifiable "valves" and LV region.
    pts = np.array(
        [
            [0.0, 0.0, 0.0],  # 0 - LV base
            [1.0, 0.0, 0.0],  # 1
            [0.5, 0.0, 0.0],  # 2 - MV (shared with LV)
            [0.5, 1.0, 0.0],  # 3 - MV
            [0.5, 0.0, 1.0],  # 4 - MV
            [2.0, 0.0, 0.0],  # 5 - TV
            [2.0, 1.0, 0.0],  # 6 - TV
            [2.0, 0.0, 1.0],  # 7 - TV
        ],
        dtype=float,
    )
    # 3 tetrahedra: LV (tag=1), MV (tag=7), TV (tag=8)
    tets = np.array(
        [
            [4, 0, 1, 2, 3],  # LV tet (vtk encoding: num_points + ids)
            [4, 2, 3, 4, 1],  # MV tet
            [4, 5, 6, 7, 2],  # TV tet
        ],
        dtype=int,
    ).flatten()
    cell_types = np.array([vtk.VTK_TETRA, vtk.VTK_TETRA, vtk.VTK_TETRA], dtype=int)
    grid = pv.UnstructuredGrid(tets, cell_types, pts)
    grid.cell_data["ID"] = np.array([1, 7, 8], dtype=int)
    return grid


def _rotate_mesh_reduce(plt_msh):
    """Return the transformed point cloud as an array."""
    return np.array(plt_msh.points, dtype=float)


CASES = [
    # ------------------------------------------------------------------
    # clean_mesh
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/clean_mesh_polydata",
        func=vtktools.clean_mesh,
        args=(_PD,),
        reduce=_pd_reduce,
        fmt="json",
    ),
    # NOTE: genericThreshold / thresholdExactValue use the old VTK API
    # (ThresholdBetween / ThresholdByUpper / ThresholdByLower) which was removed
    # in VTK >= 9.1. These functions crash in the imatools conda env.
    # They are NOT captured here — the migration task (T2b2) must update the API.
    # SKIP: genericThreshold, thresholdExactValue
    # ------------------------------------------------------------------
    # point <-> cell data converters
    # set_cell_to_point_data / setCellDataToPointData need GetPolyDataOutput
    # so they only work on polydata.
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/set_cell_to_point_data",
        func=vtktools.set_cell_to_point_data,
        args=(_PD,),
        reduce=lambda pd: {
            "n_points": pd.GetNumberOfPoints(),
            "n_cells": pd.GetNumberOfCells(),
            "has_point_scalars": pd.GetPointData().GetScalars() is not None,
        },
        fmt="json",
    ),
    CaptureCase(
        name="mesh/point_to_cell_data",
        func=vtktools.point_to_cell_data,
        args=(_PD,),
        reduce=_pd_reduce,
        fmt="json",
    ),
    CaptureCase(
        name="mesh/cell_to_point_data",
        func=vtktools.cell_to_point_data,
        args=(_PD,),
        reduce=lambda pd: {
            "n_points": pd.GetNumberOfPoints(),
            "n_cells": pd.GetNumberOfCells(),
            "has_point_scalars": pd.GetPointData().GetScalars() is not None,
        },
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # convertPointDataToNpArray / convertCellDataToNpArray
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/convertPointDataToNpArray",
        func=vtktools.convertPointDataToNpArray,
        args=(_PD, "scalars"),
        fmt="npy",
    ),
    CaptureCase(
        name="mesh/convertCellDataToNpArray",
        func=vtktools.convertCellDataToNpArray,
        args=(_PD, "scalars"),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # set_vtk_scalars — needs a polydata with existing scalars array
    # Sets all cells to new values; unset cells get -1.
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/set_vtk_scalars_all",
        func=vtktools.set_vtk_scalars,
        args=(_PD, np.array([10.0, 20.0, 30.0, 40.0])),
        reduce=lambda pd: vtknp.vtk_to_numpy(pd.GetCellData().GetScalars()).tolist(),
        fmt="json",
    ),
    CaptureCase(
        name="mesh/set_vtk_scalars_indexed",
        func=vtktools.set_vtk_scalars,
        args=(_PD, np.array([99.0]), np.array([2])),
        reduce=lambda pd: vtknp.vtk_to_numpy(pd.GetCellData().GetScalars()).tolist(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # set_cell_scalars
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/set_cell_scalars",
        func=vtktools.set_cell_scalars,
        args=(_PD, 5.0),
        reduce=lambda pd: vtknp.vtk_to_numpy(pd.GetCellData().GetScalars()).tolist(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # indices_at_scalar
    # _PD has cell scalars [1.0, 2.0, 3.0, 4.0]
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/indices_at_scalar_default",
        func=vtktools.indices_at_scalar,
        args=(_PD,),
        reduce=lambda arr: arr.tolist(),
        fmt="json",
    ),
    CaptureCase(
        name="mesh/indices_at_scalar_value",
        func=vtktools.indices_at_scalar,
        args=(_PD, 2.0),
        reduce=lambda arr: arr.tolist(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # getSurfaceArea
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/getSurfaceArea",
        func=vtktools.getSurfaceArea,
        args=(_PD,),
        reduce=float,
        fmt="json",
    ),
    # NOTE: getElemPermutation is an unfinished stub in master. For same-size inputs
    # it calls np.zeros(n0, 1) which raises TypeError in numpy >= 1.24 (second arg
    # must be dtype, not int). For different sizes it returns -1 correctly.
    # Only the diff-size path is capturable; same-size crashes.
    CaptureCase(
        name="mesh/getElemPermutation_diff_size",
        func=vtktools.getElemPermutation,
        args=(np.array([[0, 1, 2]]), np.array([[0, 1, 2], [1, 2, 3]])),
        reduce=lambda r: int(r) if r is not None else None,
        fmt="json",
    ),
    # NOTE: getElemPermutation_same_size skipped — crashes due to np.zeros(n0, 1)
    # stub bug in master. T2b2 must fix this during migration.
    # ------------------------------------------------------------------
    # fibrosis_score* — msh must have cell scalars
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/fibrosis_score_cell",
        func=vtktools.fibrosis_score,
        args=(_FIBROSIS_MSH0, _FIBROSIS_TH),
        kwargs={"type": "cell"},
        reduce=float,
        fmt="json",
    ),
    CaptureCase(
        name="mesh/fibrosis_score_cell_wrapper",
        func=vtktools.fibrosis_score_cell,
        args=(_FIBROSIS_MSH0, _FIBROSIS_TH),
        reduce=float,
        fmt="json",
    ),
    CaptureCase(
        name="mesh/fibrorisScore_legacy",
        func=vtktools.fibrorisScore,
        args=(_FIBROSIS_MSH0, _FIBROSIS_TH),
        reduce=float,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # fibrosis_overlap* — needs two meshes with same cell count and scalars > 0
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/fibrosis_overlap_cell",
        func=vtktools.fibrosis_overlap,
        args=(_FIBROSIS_MSH0, _FIBROSIS_MSH1, _FIBROSIS_TH),
        kwargs={"type": "cell"},
        reduce=_fibrosis_count_reduce,
        fmt="json",
    ),
    CaptureCase(
        name="mesh/fibrosis_overlap_cells_wrapper",
        func=vtktools.fibrosis_overlap_cells,
        args=(_FIBROSIS_MSH0, _FIBROSIS_MSH1, _FIBROSIS_TH),
        reduce=_fibrosis_count_reduce,
        fmt="json",
    ),
    CaptureCase(
        name="mesh/fibrosisOverlapCell_legacy",
        func=vtktools.fibrosisOverlapCell,
        args=(_FIBROSIS_MSH0, _FIBROSIS_MSH1, _FIBROSIS_TH),
        reduce=_fibrosis_count_reduce,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # tag_elements_by_voxel_boxes — adds 'scar' array to unstructured grid
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/tag_elements_by_voxel_boxes",
        func=vtktools.tag_elements_by_voxel_boxes,
        args=(_UG, _VOXEL_BOXES),
        reduce=_ug_scar_reduce("scar"),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # tag_mesh_elements_by_voxel_boxes — different signature (centroids + boxes)
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/tag_mesh_elements_by_voxel_boxes_centroids",
        func=vtktools.tag_mesh_elements_by_voxel_boxes,
        args=(_UG, np.array([_UG_COG0, _UG_COG1]), _VOXEL_BOXES),
        reduce=_ug_scar_reduce("scar"),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # tag_mesh_elements_by_growing_from_seed (basic BFS)
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/tag_mesh_elements_by_growing_from_seed",
        func=vtktools.tag_mesh_elements_by_growing_from_seed,
        args=(_UG, _SEED_POINTS, _VOXEL_BOXES),
        reduce=_ug_scar_reduce("scar"),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # tag_mesh_elements_by_growing_from_seed_optimized (master's NEW signature
    # includes label_value — RECONCILIATION NOTE)
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/tag_mesh_elements_by_growing_from_seed_optimized",
        func=vtktools.tag_mesh_elements_by_growing_from_seed_optimized,
        args=(_UG, _SEED_POINTS, _VOXEL_BOXES),
        kwargs={"label_value": 2},
        reduce=_ug_scar_reduce("scar"),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # tag_mesh_elements_parallel_regions
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/tag_mesh_elements_parallel_regions",
        func=vtktools.tag_mesh_elements_parallel_regions,
        args=(_UG, _SEED_POINTS, _VOXEL_BOXES),
        reduce=_ug_scar_reduce("scar"),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # flip_xy — modifies polydata in place, returns None; check transformed pts
    # We deep-copy to avoid mutating shared fixture.
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/flip_xy",
        func=lambda: _flip_xy_capture(),
        args=(),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # global_centre_of_mass
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/global_centre_of_mass",
        func=vtktools.global_centre_of_mass,
        args=(_PD,),
        reduce=lambda arr: arr.tolist(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # translate_to_point
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/translate_to_point_origin",
        func=vtktools.translate_to_point,
        args=(_PD,),
        reduce=lambda pd: {
            "cog": vtktools.global_centre_of_mass(pd).tolist(),
            "n_points": pd.GetNumberOfPoints(),
        },
        fmt="json",
    ),
    CaptureCase(
        name="mesh/translate_to_point_custom",
        func=vtktools.translate_to_point,
        args=(_PD, [1.0, 2.0, 3.0]),
        reduce=lambda pd: {
            "cog": vtktools.global_centre_of_mass(pd).tolist(),
            "n_points": pd.GetNumberOfPoints(),
        },
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # join_vtk
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/join_vtk",
        func=vtktools.join_vtk,
        args=(_PD, _PD),
        reduce=lambda pd: {
            "n_points": pd.GetNumberOfPoints(),
            "n_cells": pd.GetNumberOfCells(),
        },
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # np_to_vtk_array
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/np_to_vtk_array",
        func=vtktools.np_to_vtk_array,
        args=(np.array([1.0, 2.0, 3.0]), "test_field"),
        reduce=lambda arr: {
            "name": arr.GetName(),
            "values": vtknp.vtk_to_numpy(arr).tolist(),
        },
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # mask_cell_scalars
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/mask_cell_scalars",
        func=vtktools.mask_cell_scalars,
        args=(_PD, [99.0], [1]),
        reduce=lambda pd: vtknp.vtk_to_numpy(pd.GetCellData().GetScalars()).tolist(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # ugrid2polydata
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/ugrid2polydata",
        func=vtktools.ugrid2polydata,
        args=(_UG,),
        reduce=_pd_reduce,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # cogs_from_ugrid
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/cogs_from_ugrid",
        func=vtktools.cogs_from_ugrid,
        args=(_UG,),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # extractPointsAndElemsFromVtk — returns (Xpts, Tri) for triangular mesh
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/extractPointsAndElemsFromVtk_pts",
        func=lambda: vtktools.extractPointsAndElemsFromVtk(_PD)[0],
        args=(),
        fmt="npy",
    ),
    CaptureCase(
        name="mesh/extractPointsAndElemsFromVtk_tri",
        func=lambda: vtktools.extractPointsAndElemsFromVtk(_PD)[1],
        args=(),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # get_element_cogs — similar to get_cog_per_element but different impl
    # Works on any mesh type (uses GetCell instead of extractPointsAndElemsFromVtk)
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/get_element_cogs_polydata",
        func=vtktools.get_element_cogs,
        args=(_PD,),
        fmt="npy",
    ),
    CaptureCase(
        name="mesh/get_element_cogs_ugrid",
        func=vtktools.get_element_cogs,
        args=(_UG,),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # compare_mesh_sizes — reads files from disk; captured via wrapper
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/compare_mesh_sizes_pts",
        func=_compare_mesh_sizes_pts,
        args=(),
        reduce=_compare_mesh_sizes_reduce,
        fmt="json",
    ),
    CaptureCase(
        name="mesh/compare_mesh_sizes_elem",
        func=_compare_mesh_sizes_elem,
        args=(),
        reduce=_compare_mesh_sizes_reduce,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # map_cells — takes msh_large, cog_small, tot_small, large_id, small_id
    # We use polydata as the "large" mesh, and two COG seed points as "small".
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/map_cells",
        func=vtktools.map_cells,
        args=(
            _PD,
            np.array([_UG_COG0, _UG_COG1]),
            2,
            "LARGE",
            "SMALL",
        ),
        reduce=_map_reduce,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # map_points — takes msh_large, msh_small, large_id, small_id
    # Use the same polydata for both (self-mapping → distances should be 0).
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/map_points",
        func=vtktools.map_points,
        args=(_PD, _PD, "LARGE", "SMALL"),
        reduce=_map_reduce,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # verify_cell_indices
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/verify_cell_indices",
        func=vtktools.verify_cell_indices,
        args=(
            _PD,
            np.array([0, 1]),
            vtktools.get_element_cogs(_PD)[[0, 1]],
        ),
        reduce=float,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # rotate_mesh (from utils.py) — complex fixture built inline
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh/rotate_mesh",
        func=common_utils.rotate_mesh,
        args=(_make_rotate_mesh_input(),),
        reduce=_rotate_mesh_reduce,
        fmt="npy",
    ),
]


# ---------------------------------------------------------------------------
# flip_xy capture helper (function modifies in place, returns None)
# ---------------------------------------------------------------------------


def _flip_xy_capture():
    """Deep-copy polydata, apply flip_xy, return point array."""
    pd_copy = vtk.vtkPolyData()
    pd_copy.DeepCopy(_PD)
    vtktools.flip_xy(pd_copy)
    pts = np.array([pd_copy.GetPoint(i) for i in range(pd_copy.GetNumberOfPoints())])
    return pts
