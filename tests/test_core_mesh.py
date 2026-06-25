"""Characterization tests for ``imatools.core.mesh`` (T1e).

All tests import from the TARGET location ``imatools.core.mesh`` or
``imatools.core.geometry`` (for ``rotate_mesh`` from utils).  Those modules do not
exist yet — they will be created by migration task T2b2.  Until then every test is
marked ``xfail(strict=False)`` so it is collected but does not block CI.

Functions characterized here (from master ``imatools/common/vtktools.py``):
  ``clean_mesh``,
  ``set_cell_to_point_data``, ``setCellDataToPointData``,
  ``point_to_cell_data``, ``cell_to_point_data``,
  ``convertPointDataToNpArray``, ``convertCellDataToNpArray``,
  ``set_vtk_scalars``, ``set_cell_scalars``, ``indices_at_scalar``,
  ``getSurfaceArea``, ``getElemPermutation``,
  ``fibrosis_score``, ``fibrosis_score_cell``, ``fibrosis_score_point``,
  ``fibrorisScore``,
  ``fibrosis_overlap``, ``fibrosis_overlap_points``, ``fibrosis_overlap_cells``,
  ``fibrosisOverlapCell``,
  ``tag_elements_by_voxel_boxes``, ``tag_mesh_elements_by_voxel_boxes``,
  ``tag_mesh_elements_by_growing_from_seed``,
  ``tag_mesh_elements_by_growing_from_seed_optimized`` (+``label_value`` param),
  ``tag_mesh_elements_parallel_regions``,
  ``flip_xy``, ``global_centre_of_mass``, ``translate_to_point``, ``join_vtk``,
  ``np_to_vtk_array``, ``mask_cell_scalars``,
  ``ugrid2polydata``, ``cogs_from_ugrid``, ``extractPointsAndElemsFromVtk``,
  ``get_element_cogs``, ``compare_mesh_sizes``, ``map_cells``, ``map_points``,
  ``verify_cell_indices``.
From master ``imatools/common/utils.py``:
  ``rotate_mesh``.

SKIPPED (master bugs incompatible with installed VTK/numpy):
  ``genericThreshold`` / ``thresholdExactValue`` — uses deprecated
  ``ThresholdBetween``/``ThresholdByUpper``/``ThresholdByLower`` API removed in VTK >= 9.1.
  ``getElemPermutation`` (same-size path) — crashes with ``np.zeros(n0, 1)`` stub bug.

Golden values were captured from master via::

    M=~/dev/python/imatools.worktrees/master
    ~/opt/anaconda3/bin/conda run -n imatools env PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module mesh --out tests/golden
"""

from __future__ import annotations

import os
import tempfile

import _fixtures as fx
import numpy as np
import pytest
import vtk
import vtk.util.numpy_support as vtknp

# ---------------------------------------------------------------------------
# Shared fixture-derived constants (mirror _golden_cases/mesh.py)
# ---------------------------------------------------------------------------

_PD = fx.polydata()
_UG = fx.unstructured_grid()


def _make_scored_polydata():
    """Triangulated polydata with float cell scalars (no builtin scalars)."""
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
    arr = vtk.vtkFloatArray()
    arr.SetName("scalars")
    for v in values:
        arr.InsertNextValue(v)
    pd.GetCellData().SetScalars(arr)
    return pd


_FIBROSIS_MSH0 = _attach_cell_scalars(_make_scored_polydata(), [0.3, 0.6, 0.9, 0.4])
_FIBROSIS_MSH1 = _attach_cell_scalars(_make_scored_polydata(), [0.2, 0.7, 0.5, 0.8])
_FIBROSIS_TH = 0.5


def _ug_cog(ug, cell_id):
    cell = ug.GetCell(cell_id)
    n = cell.GetNumberOfPoints()
    cog = np.zeros(3)
    for i in range(n):
        cog += np.array(ug.GetPoint(cell.GetPointId(i)))
    return cog / n


_UG_COG0 = _ug_cog(_UG, 0)
_UG_COG1 = _ug_cog(_UG, 1)

_EPS = 0.3


def _box_around(cog, eps=_EPS):
    c = np.array(cog)
    return np.array(
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


_SEED_POINTS = np.array([_UG_COG0, _UG_COG1])
_VOXEL_BOXES = [_box_around(_UG_COG0), _box_around(_UG_COG1)]


def _make_rotate_mesh_input():
    import pyvista as pv

    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    tets = np.array(
        [
            [4, 0, 1, 2, 3],
            [4, 2, 3, 4, 1],
            [4, 5, 6, 7, 2],
        ],
        dtype=int,
    ).flatten()
    cell_types = np.array([vtk.VTK_TETRA, vtk.VTK_TETRA, vtk.VTK_TETRA], dtype=int)
    grid = pv.UnstructuredGrid(tets, cell_types, pts)
    grid.cell_data["ID"] = np.array([1, 7, 8], dtype=int)
    return grid


# ---------------------------------------------------------------------------
# clean_mesh
# ---------------------------------------------------------------------------


def test_clean_mesh_polydata(golden):
    from imatools.core.mesh import clean_mesh

    result = clean_mesh(_PD)
    expected = golden("mesh/clean_mesh_polydata")
    assert result.GetNumberOfPoints() == expected["n_points"]
    assert result.GetNumberOfCells() == expected["n_cells"]


# ---------------------------------------------------------------------------
# set_cell_to_point_data
# ---------------------------------------------------------------------------


def test_set_cell_to_point_data(golden):
    from imatools.core.mesh import set_cell_to_point_data

    result = set_cell_to_point_data(_PD)
    expected = golden("mesh/set_cell_to_point_data")
    assert result.GetNumberOfPoints() == expected["n_points"]
    assert result.GetNumberOfCells() == expected["n_cells"]
    assert (result.GetPointData().GetScalars() is not None) == expected["has_point_scalars"]


# ---------------------------------------------------------------------------
# point_to_cell_data
# ---------------------------------------------------------------------------


def test_point_to_cell_data(golden):
    from imatools.core.mesh import point_to_cell_data

    result = point_to_cell_data(_PD)
    expected = golden("mesh/point_to_cell_data")
    assert result.GetNumberOfPoints() == expected["n_points"]
    assert result.GetNumberOfCells() == expected["n_cells"]


# ---------------------------------------------------------------------------
# cell_to_point_data
# ---------------------------------------------------------------------------


def test_cell_to_point_data(golden):
    from imatools.core.mesh import cell_to_point_data

    result = cell_to_point_data(_PD)
    expected = golden("mesh/cell_to_point_data")
    assert result.GetNumberOfPoints() == expected["n_points"]
    assert result.GetNumberOfCells() == expected["n_cells"]
    assert (result.GetPointData().GetScalars() is not None) == expected["has_point_scalars"]


# ---------------------------------------------------------------------------
# convertPointDataToNpArray / convertCellDataToNpArray
# ---------------------------------------------------------------------------


def test_convert_point_data_to_np_array(golden):
    from imatools.core.mesh import convertPointDataToNpArray  # noqa: N802

    result = convertPointDataToNpArray(_PD, "scalars")
    expected = golden("mesh/convertPointDataToNpArray")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_convert_cell_data_to_np_array(golden):
    from imatools.core.mesh import convertCellDataToNpArray  # noqa: N802

    result = convertCellDataToNpArray(_PD, "scalars")
    expected = golden("mesh/convertCellDataToNpArray")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# set_vtk_scalars
# ---------------------------------------------------------------------------


def test_set_vtk_scalars_all(golden):
    from imatools.core.mesh import set_vtk_scalars

    result = set_vtk_scalars(_PD, np.array([10.0, 20.0, 30.0, 40.0]))
    expected = golden("mesh/set_vtk_scalars_all")
    result_values = vtknp.vtk_to_numpy(result.GetCellData().GetScalars()).tolist()
    assert result_values == pytest.approx(expected, rel=1e-7)


def test_set_vtk_scalars_indexed(golden):
    from imatools.core.mesh import set_vtk_scalars

    result = set_vtk_scalars(_PD, np.array([99.0]), np.array([2]))
    expected = golden("mesh/set_vtk_scalars_indexed")
    result_values = vtknp.vtk_to_numpy(result.GetCellData().GetScalars()).tolist()
    assert result_values == pytest.approx(expected, rel=1e-7)


# ---------------------------------------------------------------------------
# set_cell_scalars
# ---------------------------------------------------------------------------


def test_set_cell_scalars(golden):
    from imatools.core.mesh import set_cell_scalars

    result = set_cell_scalars(_PD, 5.0)
    expected = golden("mesh/set_cell_scalars")
    result_values = vtknp.vtk_to_numpy(result.GetCellData().GetScalars()).tolist()
    assert result_values == pytest.approx(expected, rel=1e-7)


# ---------------------------------------------------------------------------
# indices_at_scalar
# ---------------------------------------------------------------------------


def test_indices_at_scalar_default(golden):
    from imatools.core.mesh import indices_at_scalar

    result = indices_at_scalar(_PD)
    expected = golden("mesh/indices_at_scalar_default")
    assert result.tolist() == expected


def test_indices_at_scalar_value(golden):
    from imatools.core.mesh import indices_at_scalar

    result = indices_at_scalar(_PD, 2.0)
    expected = golden("mesh/indices_at_scalar_value")
    assert result.tolist() == expected


# ---------------------------------------------------------------------------
# getSurfaceArea
# ---------------------------------------------------------------------------


def test_get_surface_area(golden):
    from imatools.core.mesh import getSurfaceArea  # noqa: N802

    result = getSurfaceArea(_PD)
    expected = golden("mesh/getSurfaceArea")
    assert result == pytest.approx(expected, rel=1e-7)


# ---------------------------------------------------------------------------
# getElemPermutation — only diff-size path (same-size is a stub bug)
# ---------------------------------------------------------------------------


def test_get_elem_permutation_diff_size(golden):
    from imatools.core.mesh import getElemPermutation  # noqa: N802

    result = getElemPermutation(np.array([[0, 1, 2]]), np.array([[0, 1, 2], [1, 2, 3]]))
    expected = golden("mesh/getElemPermutation_diff_size")
    assert int(result) == expected


# ---------------------------------------------------------------------------
# fibrosis_score*
# ---------------------------------------------------------------------------


def test_fibrosis_score_cell(golden):
    from imatools.core.mesh import fibrosis_score

    result = fibrosis_score(_FIBROSIS_MSH0, _FIBROSIS_TH, type="cell")
    expected = golden("mesh/fibrosis_score_cell")
    assert result == pytest.approx(expected, rel=1e-7)


def test_fibrosis_score_cell_wrapper(golden):
    from imatools.core.mesh import fibrosis_score_cell

    result = fibrosis_score_cell(_FIBROSIS_MSH0, _FIBROSIS_TH)
    expected = golden("mesh/fibrosis_score_cell_wrapper")
    assert result == pytest.approx(expected, rel=1e-7)


def test_fibrois_score_legacy(golden):
    from imatools.core.mesh import fibrorisScore  # noqa: N802

    result = fibrorisScore(_FIBROSIS_MSH0, _FIBROSIS_TH)
    expected = golden("mesh/fibrorisScore_legacy")
    assert result == pytest.approx(expected, rel=1e-7)


# ---------------------------------------------------------------------------
# fibrosis_overlap*
# ---------------------------------------------------------------------------


def test_fibrosis_overlap_cell(golden):
    from imatools.core.mesh import fibrosis_overlap

    _msh, count_dic = fibrosis_overlap(_FIBROSIS_MSH0, _FIBROSIS_MSH1, _FIBROSIS_TH, type="cell")
    expected = golden("mesh/fibrosis_overlap_cell")
    assert {
        k: int(v) if isinstance(v, (int, float)) else v for k, v in count_dic.items()
    } == expected


def test_fibrosis_overlap_cells_wrapper(golden):
    from imatools.core.mesh import fibrosis_overlap_cells

    _msh, count_dic = fibrosis_overlap_cells(_FIBROSIS_MSH0, _FIBROSIS_MSH1, _FIBROSIS_TH)
    expected = golden("mesh/fibrosis_overlap_cells_wrapper")
    assert {
        k: int(v) if isinstance(v, (int, float)) else v for k, v in count_dic.items()
    } == expected


def test_fibrosis_overlap_cell_legacy(golden):
    from imatools.core.mesh import fibrosisOverlapCell  # noqa: N802

    _msh, count_dic = fibrosisOverlapCell(_FIBROSIS_MSH0, _FIBROSIS_MSH1, _FIBROSIS_TH)
    expected = golden("mesh/fibrosisOverlapCell_legacy")
    assert {
        k: int(v) if isinstance(v, (int, float)) else v for k, v in count_dic.items()
    } == expected


# ---------------------------------------------------------------------------
# tag_elements_by_voxel_boxes
# ---------------------------------------------------------------------------


def test_tag_elements_by_voxel_boxes(golden):
    from imatools.core.mesh import tag_elements_by_voxel_boxes

    result = tag_elements_by_voxel_boxes(_UG, _VOXEL_BOXES)
    expected = golden("mesh/tag_elements_by_voxel_boxes")
    arr_vtk = result.GetCellData().GetArray("scar")
    tags = vtknp.vtk_to_numpy(arr_vtk).tolist() if arr_vtk is not None else []
    assert tags == expected["tags"]
    assert result.GetNumberOfCells() == expected["n_cells"]


# ---------------------------------------------------------------------------
# tag_mesh_elements_by_voxel_boxes
# ---------------------------------------------------------------------------


def test_tag_mesh_elements_by_voxel_boxes_centroids(golden):
    from imatools.core.mesh import tag_mesh_elements_by_voxel_boxes

    result = tag_mesh_elements_by_voxel_boxes(_UG, np.array([_UG_COG0, _UG_COG1]), _VOXEL_BOXES)
    expected = golden("mesh/tag_mesh_elements_by_voxel_boxes_centroids")
    arr_vtk = result.GetCellData().GetArray("scar")
    tags = vtknp.vtk_to_numpy(arr_vtk).tolist() if arr_vtk is not None else []
    assert tags == expected["tags"]


# ---------------------------------------------------------------------------
# tag_mesh_elements_by_growing_from_seed
# ---------------------------------------------------------------------------


def test_tag_mesh_elements_by_growing_from_seed(golden):
    from imatools.core.mesh import tag_mesh_elements_by_growing_from_seed

    result = tag_mesh_elements_by_growing_from_seed(_UG, _SEED_POINTS, _VOXEL_BOXES)
    expected = golden("mesh/tag_mesh_elements_by_growing_from_seed")
    arr_vtk = result.GetCellData().GetArray("scar")
    tags = vtknp.vtk_to_numpy(arr_vtk).tolist() if arr_vtk is not None else []
    assert tags == expected["tags"]
    assert result.GetNumberOfCells() == expected["n_cells"]


# ---------------------------------------------------------------------------
# tag_mesh_elements_by_growing_from_seed_optimized (master's NEW signature)
# ---------------------------------------------------------------------------


def test_tag_mesh_elements_by_growing_from_seed_optimized(golden):
    from imatools.core.mesh import tag_mesh_elements_by_growing_from_seed_optimized

    result = tag_mesh_elements_by_growing_from_seed_optimized(
        _UG, _SEED_POINTS, _VOXEL_BOXES, label_value=2
    )
    expected = golden("mesh/tag_mesh_elements_by_growing_from_seed_optimized")
    arr_vtk = result.GetCellData().GetArray("scar")
    tags = vtknp.vtk_to_numpy(arr_vtk).tolist() if arr_vtk is not None else []
    assert tags == expected["tags"]
    assert result.GetNumberOfCells() == expected["n_cells"]


# ---------------------------------------------------------------------------
# tag_mesh_elements_parallel_regions
# ---------------------------------------------------------------------------


def test_tag_mesh_elements_parallel_regions(golden):
    from imatools.core.mesh import tag_mesh_elements_parallel_regions

    result = tag_mesh_elements_parallel_regions(_UG, _SEED_POINTS, _VOXEL_BOXES)
    expected = golden("mesh/tag_mesh_elements_parallel_regions")
    arr_vtk = result.GetCellData().GetArray("scar")
    tags = vtknp.vtk_to_numpy(arr_vtk).tolist() if arr_vtk is not None else []
    assert tags == expected["tags"]


# ---------------------------------------------------------------------------
# flip_xy
# ---------------------------------------------------------------------------


def test_flip_xy(golden):
    from imatools.core.mesh import flip_xy

    pd_copy = vtk.vtkPolyData()
    pd_copy.DeepCopy(_PD)
    flip_xy(pd_copy)
    pts = np.array([pd_copy.GetPoint(i) for i in range(pd_copy.GetNumberOfPoints())])
    expected = golden("mesh/flip_xy")
    np.testing.assert_allclose(pts, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# global_centre_of_mass
# ---------------------------------------------------------------------------


def test_global_centre_of_mass(golden):
    from imatools.core.mesh import global_centre_of_mass

    result = global_centre_of_mass(_PD)
    expected = golden("mesh/global_centre_of_mass")
    assert result.tolist() == pytest.approx(expected, rel=1e-7)


# ---------------------------------------------------------------------------
# translate_to_point
# ---------------------------------------------------------------------------


def test_translate_to_point_origin(golden):
    from imatools.core.mesh import global_centre_of_mass, translate_to_point

    result = translate_to_point(_PD)
    expected = golden("mesh/translate_to_point_origin")
    assert global_centre_of_mass(result).tolist() == pytest.approx(expected["cog"], abs=1e-7)
    assert result.GetNumberOfPoints() == expected["n_points"]


def test_translate_to_point_custom(golden):
    from imatools.core.mesh import global_centre_of_mass, translate_to_point

    result = translate_to_point(_PD, [1.0, 2.0, 3.0])
    expected = golden("mesh/translate_to_point_custom")
    assert global_centre_of_mass(result).tolist() == pytest.approx(expected["cog"], rel=1e-7)
    assert result.GetNumberOfPoints() == expected["n_points"]


# ---------------------------------------------------------------------------
# join_vtk
# ---------------------------------------------------------------------------


def test_join_vtk(golden):
    from imatools.core.mesh import join_vtk

    result = join_vtk(_PD, _PD)
    expected = golden("mesh/join_vtk")
    assert result.GetNumberOfPoints() == expected["n_points"]
    assert result.GetNumberOfCells() == expected["n_cells"]


# ---------------------------------------------------------------------------
# np_to_vtk_array
# ---------------------------------------------------------------------------


def test_np_to_vtk_array(golden):
    from imatools.core.mesh import np_to_vtk_array

    result = np_to_vtk_array(np.array([1.0, 2.0, 3.0]), "test_field")
    expected = golden("mesh/np_to_vtk_array")
    assert result.GetName() == expected["name"]
    assert vtknp.vtk_to_numpy(result).tolist() == pytest.approx(expected["values"], rel=1e-7)


# ---------------------------------------------------------------------------
# mask_cell_scalars
# ---------------------------------------------------------------------------


def test_mask_cell_scalars(golden):
    from imatools.core.mesh import mask_cell_scalars

    result = mask_cell_scalars(_PD, [99.0], [1])
    expected = golden("mesh/mask_cell_scalars")
    result_values = vtknp.vtk_to_numpy(result.GetCellData().GetScalars()).tolist()
    assert result_values == pytest.approx(expected, rel=1e-7)


# ---------------------------------------------------------------------------
# ugrid2polydata
# ---------------------------------------------------------------------------


def test_ugrid2polydata(golden):
    from imatools.core.mesh import ugrid2polydata

    result = ugrid2polydata(_UG)
    expected = golden("mesh/ugrid2polydata")
    assert result.GetNumberOfPoints() == expected["n_points"]
    assert result.GetNumberOfCells() == expected["n_cells"]


# ---------------------------------------------------------------------------
# cogs_from_ugrid
# ---------------------------------------------------------------------------


def test_cogs_from_ugrid(golden):
    from imatools.core.mesh import cogs_from_ugrid

    result = cogs_from_ugrid(_UG)
    expected = golden("mesh/cogs_from_ugrid")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# extractPointsAndElemsFromVtk
# ---------------------------------------------------------------------------


def test_extract_points_and_elems_from_vtk_pts(golden):
    from imatools.core.mesh import extractPointsAndElemsFromVtk  # noqa: N802

    pts, _tri = extractPointsAndElemsFromVtk(_PD)
    expected = golden("mesh/extractPointsAndElemsFromVtk_pts")
    np.testing.assert_allclose(pts, expected, rtol=1e-7)


def test_extract_points_and_elems_from_vtk_tri(golden):
    from imatools.core.mesh import extractPointsAndElemsFromVtk  # noqa: N802

    _pts, tri = extractPointsAndElemsFromVtk(_PD)
    expected = golden("mesh/extractPointsAndElemsFromVtk_tri")
    np.testing.assert_array_equal(tri, expected)


# ---------------------------------------------------------------------------
# get_element_cogs
# ---------------------------------------------------------------------------


def test_get_element_cogs_polydata(golden):
    from imatools.core.mesh import get_element_cogs

    result = get_element_cogs(_PD)
    expected = golden("mesh/get_element_cogs_polydata")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_get_element_cogs_ugrid(golden):
    from imatools.core.mesh import get_element_cogs

    result = get_element_cogs(_UG)
    expected = golden("mesh/get_element_cogs_ugrid")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# compare_mesh_sizes — uses temp files
# ---------------------------------------------------------------------------


def _build_small_pd():
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
    return pd2


def test_compare_mesh_sizes_pts(golden):
    from imatools.core.mesh import compare_mesh_sizes, write_vtk

    expected = golden("mesh/compare_mesh_sizes_pts")
    with tempfile.TemporaryDirectory() as tmpdir:
        write_vtk(_PD, tmpdir, "a", output_type="polydata")
        write_vtk(_build_small_pd(), tmpdir, "b", output_type="polydata")
        path_a = os.path.join(tmpdir, "a.vtk")
        path_b = os.path.join(tmpdir, "b.vtk")
        result = compare_mesh_sizes(path_a, path_b, "A", "B", 0)
    assert int(result[2]) == expected["tot_large"]
    assert int(result[3]) == expected["tot_small"]
    assert result[4] == expected["large_id"]
    assert result[5] == expected["small_id"]


def test_compare_mesh_sizes_elem(golden):
    from imatools.core.mesh import compare_mesh_sizes, write_vtk

    expected = golden("mesh/compare_mesh_sizes_elem")
    with tempfile.TemporaryDirectory() as tmpdir:
        write_vtk(_PD, tmpdir, "a", output_type="polydata")
        write_vtk(_build_small_pd(), tmpdir, "b", output_type="polydata")
        path_a = os.path.join(tmpdir, "a.vtk")
        path_b = os.path.join(tmpdir, "b.vtk")
        result = compare_mesh_sizes(path_a, path_b, "A", "B", 1)
    assert int(result[2]) == expected["tot_large"]
    assert int(result[3]) == expected["tot_small"]
    assert result[4] == expected["large_id"]
    assert result[5] == expected["small_id"]


# ---------------------------------------------------------------------------
# map_cells
# ---------------------------------------------------------------------------


def test_map_cells(golden):
    from imatools.core.mesh import map_cells

    cog_small = np.array([_UG_COG0, _UG_COG1])
    result = map_cells(_PD, cog_small, 2, "LARGE", "SMALL")
    expected = golden("mesh/map_cells")
    # Compare index arrays
    np.testing.assert_array_equal(result["SMALL"], expected["SMALL"])
    np.testing.assert_array_equal(result["LARGE"], expected["LARGE"])


# ---------------------------------------------------------------------------
# map_points
# ---------------------------------------------------------------------------


def test_map_points(golden):
    from imatools.core.mesh import map_points

    result = map_points(_PD, _PD, "LARGE", "SMALL")
    expected = golden("mesh/map_points")
    np.testing.assert_array_equal(result["SMALL"], expected["SMALL"])
    np.testing.assert_array_equal(result["LARGE"], expected["LARGE"])


# ---------------------------------------------------------------------------
# verify_cell_indices
# ---------------------------------------------------------------------------


def test_verify_cell_indices(golden):
    from imatools.core.mesh import get_element_cogs, verify_cell_indices

    cogs = get_element_cogs(_PD)
    result = verify_cell_indices(_PD, np.array([0, 1]), cogs[[0, 1]])
    expected = golden("mesh/verify_cell_indices")
    assert float(result) == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# rotate_mesh (from utils.py — target: imatools.core.mesh)
# ---------------------------------------------------------------------------


def test_rotate_mesh(golden):
    from imatools.core.mesh import rotate_mesh

    grid = _make_rotate_mesh_input()
    result = rotate_mesh(grid)
    expected = golden("mesh/rotate_mesh")
    # atol absorbs near-zero floating-point noise (~1e-16) from the rotation so the
    # comparison stays valid against the MASTER golden across numpy versions
    # (rtol alone fails on coordinates that should be ~0). Keeps master as the oracle.
    np.testing.assert_allclose(np.array(result.points), expected, rtol=1e-6, atol=1e-9)
