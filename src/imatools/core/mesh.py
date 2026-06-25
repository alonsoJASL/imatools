# src/imatools/core/mesh.py
"""Mesh transform functions migrated from ``imatools.common.vtktools`` and
``imatools.common.utils`` (T2b2).

The public functions here are the authoritative implementations.  The old
modules re-export them via bottom-of-file shims so all existing import paths
keep working unchanged.

Cat-A fixes applied (not validated against master output because master crashes
in the installed VTK / numpy environment):

1. **VTK >= 9.1 threshold API** — ``genericThreshold`` / ``thresholdExactValue``
   used the removed ``ThresholdBetween`` / ``ThresholdByUpper`` /
   ``ThresholdByLower`` methods.  Replaced with the VTK >= 9.1 API:
   ``SetLowerThreshold`` / ``SetUpperThreshold`` +
   ``SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN /
   THRESHOLD_UPPER / THRESHOLD_LOWER)``.

2. **``getElemPermutation`` numpy dtype** — line ``np.zeros(n0, 1)`` passed
   ``1`` as the dtype argument, which raises ``TypeError`` on numpy >= 1.24.
   Changed to ``np.zeros(n0, dtype=int)``.  Only the diff-size path (returns
   ``-1``) has a golden; the same-size path (returns ``0``) has NO golden and
   remains an incomplete stub — the fix is UNVALIDATED against master output.

Uncovered wrappers moved verbatim:
- ``fibrosis_score_point`` — no golden (requires point-scalar mesh); moved for
  completeness, tested only structurally.
- ``fibrosis_overlap_points`` — same situation; moved for completeness.
"""

from __future__ import annotations

import copy
import math
import sys
from collections import deque
from typing import List, Optional

import numpy as np
import vtk
import vtk.util.numpy_support as vtknp

from imatools.common.config import configure_logging
from imatools.core.geometry import (
    get_cog_per_element,
    l2_norm,
    point_in_aabb,
    precompute_valid_cells,
    rotation_matrix,
)

logger = configure_logging(__name__)


# ---------------------------------------------------------------------------
# Lazy accessor for vtktools — avoids circular import at module-load time.
# vtktools imports this module at its bottom (shim), so we must NOT import
# vtktools at our top level.  Use _vtk().<name> for helpers that stay there.
# ---------------------------------------------------------------------------
def _vtk():
    """Return the vtktools module (already loaded when any mesh fn is called)."""
    import imatools.common.vtktools as _m  # noqa: PLC0415

    return _m


# ---------------------------------------------------------------------------
# Private helpers (stay internal; support the tag_* functions below)
# ---------------------------------------------------------------------------


def _build_adjacency_list(vtk_mesh):
    adjacency = [[] for _ in range(vtk_mesh.GetNumberOfCells())]
    vtk_mesh.BuildLinks()

    for cell_id in range(vtk_mesh.GetNumberOfCells()):
        cell_point_ids = vtk_mesh.GetCell(cell_id).GetPointIds()
        num_points = cell_point_ids.GetNumberOfIds()

        for i in range(num_points):
            pt_id = cell_point_ids.GetId(i)
            cell_ids = vtk.vtkIdList()
            vtk_mesh.GetPointCells(pt_id, cell_ids)

            for j in range(cell_ids.GetNumberOfIds()):
                neighbor_cell_id = cell_ids.GetId(j)
                if neighbor_cell_id != cell_id:
                    adjacency[cell_id].append(neighbor_cell_id)
    return adjacency


def _build_adjacency_list_optimized(msh) -> List[List[int]]:
    """Optimized adjacency list building with better memory usage."""
    num_cells = msh.GetNumberOfCells()
    adjacency = [[] for _ in range(num_cells)]

    for cell_id in range(num_cells):
        cell = msh.GetCell(cell_id)
        point_ids = []

        for i in range(cell.GetNumberOfPoints()):
            point_ids.append(cell.GetPointId(i))

        neighbor_set = set()
        for point_id in point_ids:
            point_cells = vtk.vtkIdList()
            msh.GetPointCells(point_id, point_cells)

            for j in range(point_cells.GetNumberOfIds()):
                neighbor_id = point_cells.GetId(j)
                if neighbor_id != cell_id:
                    neighbor_set.add(neighbor_id)

        adjacency[cell_id] = list(neighbor_set)

    return adjacency


def _seed_cell_field(msh, label_name: str, num_cells: int) -> np.ndarray:
    """Return a writable working copy of the named cell field for overlay.

    If a cell array called ``label_name`` already exists (and has the right
    length) its values are preserved so tagging overlays onto it. Otherwise a
    fresh int32 zero array is created.
    """
    existing = msh.GetCellData().GetArray(label_name)
    if existing is not None and existing.GetNumberOfTuples() == num_cells:
        logger.info(f"Overlaying onto existing cell field '{label_name}'")
        return vtknp.vtk_to_numpy(existing).copy()

    if existing is not None:
        logger.warning(
            f"Existing field '{label_name}' has {existing.GetNumberOfTuples()} "
            f"tuples but mesh has {num_cells} cells; ignoring it and creating a new field."
        )
    return np.zeros(num_cells, dtype=np.int32)


def _attach_cell_field(msh, values: np.ndarray, label_name: str) -> None:
    """Write ``values`` back onto the mesh as the named cell field.

    Preserves the numpy dtype of ``values`` (so an existing float field stays
    float). ``AddArray`` replaces any same-named array.
    """
    vtk_array = vtknp.numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    vtk_array.SetName(label_name)
    msh.GetCellData().AddArray(vtk_array)


# ---------------------------------------------------------------------------
# VTK data-type conversions
# ---------------------------------------------------------------------------


def set_cell_to_point_data(msh, fieldname="scalars"):
    c2pt = vtk.vtkCellDataToPointData()
    c2pt.SetInputData(msh)
    c2pt.PassCellDataOn()
    c2pt.SetContributingCellOption(0)
    c2pt.Update()

    omsh = c2pt.GetPolyDataOutput()
    omsh.GetPointData().GetScalars().SetName(fieldname)

    return omsh


def setCellDataToPointData(msh, fieldname="scalars"):  # noqa: N802
    """Legacy name — use set_cell_to_point_data instead."""
    print(__doc__)
    return set_cell_to_point_data(msh, fieldname)


def point_to_cell_data(msh, fieldname="scalars"):
    """Convert point data to cell data."""
    p2c = vtk.vtkPointDataToCellData()
    p2c.SetInputData(msh)
    p2c.PassPointDataOn()
    p2c.Update()

    omsh = p2c.GetOutput()
    omsh.GetCellData().GetScalars().SetName(fieldname)

    return omsh


def cell_to_point_data(msh, fieldname="scalars"):
    """Convert cell data to point data."""
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(msh)
    c2p.PassCellDataOn()
    c2p.Update()

    omsh = c2p.GetOutput()
    omsh.GetPointData().GetScalars().SetName(fieldname)

    return omsh


def convertPointDataToNpArray(vtk_input, str_scalars):  # noqa: N802
    """Convert vtk scalar data to numpy array."""
    vtkArrayDistance = vtk_input.GetPointData().GetScalars(str_scalars)  # noqa: N806
    distance = vtknp.vtk_to_numpy(vtkArrayDistance)

    return distance


def convertCellDataToNpArray(vtk_input, str_scalars):  # noqa: N802
    """Convert vtk (cell) scalar data to numpy array."""
    vtkArrayDistance = vtk_input.GetCellData().GetScalars(str_scalars)  # noqa: N806
    distance = vtknp.vtk_to_numpy(vtkArrayDistance)

    return distance


def np_to_vtk_array(data: np.ndarray, name: str) -> vtk.vtkFloatArray:
    vtk_array = vtknp.numpy_to_vtk(data)
    vtk_array.SetName(name)

    return vtk_array


# ---------------------------------------------------------------------------
# Threshold  (Cat-A fix: VTK >= 9.1 API — UNVALIDATED against master output)
# ---------------------------------------------------------------------------


def genericThreshold(msh, exactValue, typeThres="exact"):  # noqa: N802,N803
    """Threshold polydata.

    Returns an unstructured grid.

    Cat-A fix: replaced deprecated ThresholdBetween / ThresholdByUpper /
    ThresholdByLower (removed in VTK >= 9.1) with the new API using
    SetLowerThreshold / SetUpperThreshold + SetThresholdFunction.
    This fix is NOT validated against master output (master crashes in-env).
    """
    thresBehaviour = {"exact": 0, "upper": 1, "lower": 2}  # noqa: N806

    th = vtk.vtkThreshold()
    th.SetInputData(msh)

    if thresBehaviour[typeThres] == 0:  # exact
        th.SetLowerThreshold(exactValue)
        th.SetUpperThreshold(exactValue)
        th.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    elif thresBehaviour[typeThres] == 1:  # upper
        th.SetLowerThreshold(exactValue)
        th.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_UPPER)
    elif thresBehaviour[typeThres] == 2:  # lower
        th.SetUpperThreshold(exactValue)
        th.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_LOWER)
    else:
        sys.exit(-1)

    th.Update()

    return th.GetOutput()


def thresholdExactValue(msh, exactValue):  # noqa: N802,N803
    """Threshold polydata at exact value (like a tag).

    Returns an unstructured grid.
    """
    return genericThreshold(msh, exactValue, "exact")


# ---------------------------------------------------------------------------
# Element permutation  (Cat-A fix: np.zeros dtype — UNVALIDATED same-size path)
# ---------------------------------------------------------------------------


def getElemPermutation(msh0, msh1):  # noqa: N802
    """Produce perm such that msh0[perm] = msh1.

    Cat-A fix: ``np.zeros(n0, 1)`` passed ``1`` as dtype arg (TypeError on
    numpy >= 1.24). Changed to ``np.zeros(n0, dtype=int)``.
    Only the diff-size path (returns -1) has a golden; the same-size return-0
    path is an incomplete stub — the fix is UNVALIDATED against master output.
    """
    n0 = len(msh0)
    n1 = len(msh1)

    if n0 != n1:
        return -1

    perm = np.zeros(n0, dtype=int)  # noqa: F841  Cat-A fix: was np.zeros(n0, 1)
    # for el in msh0:  (stub — body not implemented in master)

    return 0


# ---------------------------------------------------------------------------
# Surface area / mesh extraction
# ---------------------------------------------------------------------------


def getSurfaceArea(msh):  # noqa: N802
    mp = vtk.vtkMassProperties()
    mp.SetInputData(msh)
    return mp.GetSurfaceArea()


def extractPointsAndElemsFromVtk(msh):  # noqa: N802
    pts = [list(msh.GetPoint(ix)) for ix in range(msh.GetNumberOfPoints())]
    el = [
        [msh.GetCell(jx).GetPointIds().GetId(ix) for ix in range(3)]
        for jx in range(msh.GetNumberOfCells())
    ]

    Xpts = np.asarray(pts)  # noqa: N806
    Tri = np.asarray(el)  # noqa: N806

    return Xpts, Tri


def ugrid2polydata(ugrid):
    """Convert unstructured grid to polydata using the geometry filter."""
    gf = vtk.vtkGeometryFilter()
    gf.SetInputData(ugrid)
    gf.Update()

    return gf.GetOutput()


def cogs_from_ugrid(msh: vtk.vtkUnstructuredGrid):
    num_elems = msh.GetNumberOfCells()
    cogs = np.empty((num_elems, 3))

    for cid in range(num_elems):
        cell = msh.GetCell(cid)
        num_pts = cell.GetNumberOfPoints()

        centroid = np.zeros(3)
        for ix in range(num_pts):
            pt = np.array(msh.GetPoint(cell.GetPointId(ix)))
            centroid += pt
        centroid /= num_pts
        cogs[cid, :] = centroid

    return cogs


def get_element_cogs(vtk_mesh):
    cogs = []
    for i in range(vtk_mesh.GetNumberOfCells()):
        cell = vtk_mesh.GetCell(i)
        pts = cell.GetPoints()
        pts_np = np.array([pts.GetPoint(j) for j in range(pts.GetNumberOfPoints())])
        cogs.append(np.mean(pts_np, axis=0))
    return np.array(cogs)


# ---------------------------------------------------------------------------
# Scalar operations
# ---------------------------------------------------------------------------


def set_vtk_scalars(msh, array, indices=None) -> vtk.vtkPolyData:
    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(msh)

    scalars = omsh.GetCellData().GetScalars()
    if scalars is None:
        scalars = vtk.vtkFloatArray()
        scalars.SetName("scalars")
        omsh.GetCellData().SetScalars(scalars)

    inav_array = -1 * np.ones_like(convertCellDataToNpArray(omsh, "scalars"))

    if indices is not None:
        inav_array[indices] = array
    else:
        inav_array = array

    omsh.GetCellData().SetScalars(np_to_vtk_array(inav_array, "scalars"))
    return omsh


def set_cell_scalars(vtklabel, label):
    """Set the cell scalars of a vtkPolyData object."""
    return set_vtk_scalars(vtklabel, np.ones(vtklabel.GetNumberOfCells()) * label)


def indices_at_scalar(msh, scalar=0.0, fieldname="scalars") -> np.ndarray:
    scalars = convertCellDataToNpArray(msh, fieldname)
    indices = np.where(scalars == scalar)[0]

    return indices


def mask_cell_scalars(msh, values, indices, scalar_field="scalars"):
    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(msh)
    array = convertCellDataToNpArray(omsh, scalar_field)

    if len(indices) != len(values):
        raise ValueError("Indices and values must have the same length")

    for ix, indx in enumerate(indices):
        array[indx] = values[ix]

    omsh.GetCellData().SetScalars(np_to_vtk_array(array, scalar_field))
    return omsh


# ---------------------------------------------------------------------------
# Mesh cleaning / joining
# ---------------------------------------------------------------------------


def clean_mesh(msh: vtk.vtkPolyData):
    """Clean a vtkPolyData object by removing duplicate points and cells."""
    cleanFilter = vtk.vtkCleanPolyData()  # noqa: N806
    cleanFilter.SetInputData(msh)
    cleanFilter.Update()

    return cleanFilter.GetOutput()


def join_vtk(msh0, msh1):
    appendFilter = vtk.vtkAppendPolyData()  # noqa: N806
    appendFilter.AddInputData(msh0)
    appendFilter.AddInputData(msh1)
    appendFilter.Update()

    return appendFilter.GetOutput()


# ---------------------------------------------------------------------------
# Spatial transforms
# ---------------------------------------------------------------------------


def flip_xy(polydata):
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()

    for i in range(num_points):
        original_coords = points.GetPoint(i)
        modified_coords = [-original_coords[0], -original_coords[1], original_coords[2]]
        points.SetPoint(i, modified_coords)

    polydata.Modified()


def global_centre_of_mass(mesh):
    """Return the centre of mass of a mesh."""
    pts, _ = extractPointsAndElemsFromVtk(mesh)
    return np.mean(pts, axis=0)


def translate_to_point(mesh, point=[0, 0, 0]):  # noqa: B006
    """Translate a mesh to a point."""
    cog = global_centre_of_mass(mesh)
    transform = vtk.vtkTransform()
    transform.Translate(point[0] - cog[0], point[1] - cog[1], point[2] - cog[2])

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(mesh)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    return transform_filter.GetOutput()


# ---------------------------------------------------------------------------
# Mesh comparison / mapping
# ---------------------------------------------------------------------------


def compare_mesh_sizes(msh_left_name, msh_right_name, left_id, right_id, map_type_id):
    msh_left = _vtk().readVtk(msh_left_name)
    msh_right = _vtk().readVtk(msh_right_name)

    if map_type_id == 1:  # elem
        tot_left = msh_left.GetNumberOfCells()
        tot_right = msh_right.GetNumberOfCells()
    elif map_type_id == 0:  # pts
        tot_left = msh_left.GetNumberOfPoints()
        tot_right = msh_right.GetNumberOfPoints()
    else:
        return None, None, None, None, None, None

    path_large = msh_left_name  # 0
    path_small = msh_right_name  # 1
    tot_large = tot_left
    tot_small = tot_right
    large_id = left_id
    small_id = right_id

    if tot_left < tot_right:
        path_large = msh_right_name
        path_small = msh_left_name
        tot_large = tot_right
        tot_small = tot_left
        large_id = right_id
        small_id = left_id

    return path_large, path_small, tot_large, tot_small, large_id, small_id


def map_cells(msh_large, cog_small, tot_small, large_id, small_id):
    cell_locate_on_large = vtk.vtkCellLocator()
    cell_locate_on_large.SetDataSet(msh_large)
    cell_locate_on_large.BuildLocator()

    cell_ids_small = np.arange(tot_small)
    cell_ids_large = np.zeros(tot_small, dtype=int)
    l2_norm_filter = np.zeros(tot_small, dtype=float)

    pts_in_large = np.zeros((tot_small, 3), dtype=float)

    for ix in range(tot_small):
        cellId_in_large = vtk.reference(0)  # noqa: N806
        c_in_large = [0.0, 0.0, 0.0]
        dist_to_large = vtk.reference(0.0)

        cell_locate_on_large.FindClosestPoint(
            cog_small[ix], c_in_large, cellId_in_large, vtk.reference(0), dist_to_large
        )
        cell_ids_large[ix] = cellId_in_large.get()

        l2_norm_filter[ix] = dist_to_large

        pts_in_large[ix, 0] = c_in_large[0]
        pts_in_large[ix, 1] = c_in_large[1]
        pts_in_large[ix, 2] = c_in_large[2]

    mapping_dictionary_cells = {
        small_id: cell_ids_small,
        large_id: cell_ids_large,
        "distance_manual": l2_norm(cog_small - pts_in_large),
        "distance_auto": l2_norm_filter,
        "X_" + small_id.lower(): cog_small[:, 0],
        "Y_" + small_id.lower(): cog_small[:, 1],
        "Z_" + small_id.lower(): cog_small[:, 2],
        "X_" + large_id.lower(): pts_in_large[:, 0],
        "Y_" + large_id.lower(): pts_in_large[:, 1],
        "Z_" + large_id.lower(): pts_in_large[:, 2],
    }

    return mapping_dictionary_cells


def map_points(msh_large, msh_small, large_id, small_id):
    tot_small = msh_small.GetNumberOfPoints()

    pts_locate_on_large = vtk.vtkPointLocator()
    pts_locate_on_large.SetDataSet(msh_large)
    pts_locate_on_large.BuildLocator()

    pts_ids_small = np.arange(tot_small)
    pts_ids_large = np.zeros(tot_small, dtype=int)

    pts_small = np.zeros((tot_small, 3), dtype=float)
    pts_large = np.zeros((tot_small, 3), dtype=float)

    for ix in range(tot_small):
        pt_small = np.asarray(msh_small.GetPoint(ix))

        ptsId_in_large = pts_locate_on_large.FindClosestPoint(pt_small)  # noqa: N806
        pts_ids_large[ix] = ptsId_in_large

        pt_large = np.asarray(msh_large.GetPoint(ptsId_in_large))

        pts_small[ix, 0] = pt_small[0]
        pts_small[ix, 1] = pt_small[1]
        pts_small[ix, 2] = pt_small[2]

        pts_large[ix, 0] = pt_large[0]
        pts_large[ix, 1] = pt_large[1]
        pts_large[ix, 2] = pt_large[2]

    mapping_dictionary_points = {
        small_id: pts_ids_small,
        large_id: pts_ids_large,
        "distance_manual": l2_norm(pts_small - pts_large),
        "X_" + small_id.lower(): pts_small[:, 0],
        "Y_" + small_id.lower(): pts_small[:, 1],
        "Z_" + small_id.lower(): pts_small[:, 2],
        "X_" + large_id.lower(): pts_large[:, 0],
        "Y_" + large_id.lower(): pts_large[:, 1],
        "Z_" + large_id.lower(): pts_large[:, 2],
    }

    return mapping_dictionary_points


def verify_cell_indices(msh, test_indices, test_locations):
    """Verify that test_indices in the mesh (msh) are the same as test_locations.

    test_locations is linked to test_indices via the centre of gravity of the
    mesh elements.
    """
    cog = get_cog_per_element(msh)
    test_cog = cog[test_indices, :]
    diff = np.linalg.norm(test_cog - test_locations, axis=1)

    return np.sum(diff)


# ---------------------------------------------------------------------------
# Fibrosis scoring
# ---------------------------------------------------------------------------


def fibrosis_score(msh, th, type="cell"):  # noqa: A002
    """Assumes the scalars in msh have been normalised."""
    assert type in ["cell", "point"], 'Argument "type" expected to be "cell" or "point"'
    if type == "cell":
        scalars = msh.GetCellData().GetScalars()
    else:
        scalars = msh.GetPointData().GetScalars()

    total_values = msh.GetNumberOfCells() if type == "cell" else msh.GetNumberOfPoints()
    countt = float(total_values)
    countfib = 0.0

    for ix in range(total_values):
        if scalars.GetTuple1(ix) == 0:
            countt -= 1.0

        elif scalars.GetTuple1(ix) >= th:
            countfib += 1.0

    return countfib / countt


def fibrosis_score_cell(msh, th):
    return fibrosis_score(msh, th, "cell")


def fibrosis_score_point(msh, th):
    """NOTE: uncovered — no golden (requires point-scalar mesh). Moved verbatim."""
    return fibrosis_score(msh, th, "point")


def fibrorisScore(msh, th):  # noqa: N802
    """Assumes the scalars in msh have been normalised."""
    scalars = msh.GetCellData().GetScalars()
    countt = float(msh.GetNumberOfCells())
    countfib = 0.0

    for ix in range(msh.GetNumberOfCells()):
        if scalars.GetTuple1(ix) == 0:
            countt -= 1.0

        elif scalars.GetTuple1(ix) >= th:
            countfib += 1.0

    return countfib / countt


# ---------------------------------------------------------------------------
# Fibrosis overlap
# ---------------------------------------------------------------------------


def fibrosis_overlap(
    msh0, msh1, th0, th1=None, name0="msh0", name1="msh1", type="cell"
):  # noqa: A002
    """Make sure msh0 aligns with msh1 in number of cells."""
    th1 = th0 if (th1 is None) else th1
    assert type in ["cell", "point"], 'Argument "type" expected to be "cell" or "point"'

    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(msh0)
    o_scalar = vtk.vtkFloatArray()
    o_scalar.SetNumberOfComponents(1)

    if type == "cell":
        scalar0 = msh0.GetCellData().GetScalars()
        scalar1 = msh1.GetCellData().GetScalars()
    else:
        scalar1 = msh1.GetPointData().GetScalars()
        scalar0 = msh0.GetPointData().GetScalars()

    countn = 0
    count0 = 0
    count1 = 0
    countb = 0

    total_values = msh0.GetNumberOfCells() if type == "cell" else msh0.GetNumberOfPoints()
    for ix in range(total_values):
        value_assigned = 0

        if scalar0.GetTuple1(ix) == 0 or scalar1.GetTuple1(ix) == 0:
            value_assigned = -1

        else:
            if scalar0.GetTuple1(ix) >= th0:
                value_assigned += 1

            if scalar1.GetTuple1(ix) >= th1:
                value_assigned += 2

            if value_assigned == 0:
                countn += 1
            elif value_assigned == 1:
                count0 += 1
            elif value_assigned == 2:
                count1 += 1
            elif value_assigned == 3:
                countb += 1

        o_scalar.InsertNextTuple1(value_assigned)

    if type == "cell":
        omsh.GetCellData().SetScalars(o_scalar)
    else:
        omsh.GetPointData().SetScalars(o_scalar)

    countt = countn + count0 + count1 + countb
    count_dic = {"total": countt, name0: count0, name1: count1, "overlap": countb, "none": countn}

    return omsh, count_dic


def fibrosis_overlap_points(msh0, msh1, th0, th1=None, name0="msh0", name1="msh1"):
    """NOTE: uncovered — no golden (requires point-scalar mesh). Moved verbatim."""
    omsh, count_dic = fibrosis_overlap(msh0, msh1, th0, th1, name0, name1, type="point")
    return omsh, count_dic


def fibrosis_overlap_cells(msh0, msh1, th0, th1=None, name0="msh0", name1="msh1"):
    omsh, count_dic = fibrosis_overlap(msh0, msh1, th0, th1, name0, name1, type="cell")
    return omsh, count_dic


def fibrosisOverlapCell(msh0, msh1, th0, th1=None, name0="msh0", name1="msh1"):  # noqa: N802
    """Make sure msh0 aligns with msh1 in number of cells."""
    th1 = th0 if (th1 is None) else th1

    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(msh0)
    o_scalar = vtk.vtkFloatArray()
    o_scalar.SetNumberOfComponents(1)

    scalar0 = msh0.GetCellData().GetScalars()
    scalar1 = msh1.GetCellData().GetScalars()

    countn = 0
    count0 = 0
    count1 = 0
    countb = 0

    for ix in range(msh0.GetNumberOfCells()):
        value_assigned = 0

        if scalar0.GetTuple1(ix) == 0 or scalar1.GetTuple1(ix) == 0:
            value_assigned = -1

        else:
            if scalar0.GetTuple1(ix) >= th0:
                value_assigned += 1

            if scalar1.GetTuple1(ix) >= th1:
                value_assigned += 2

            if value_assigned == 0:
                countn += 1
            elif value_assigned == 1:
                count0 += 1
            elif value_assigned == 2:
                count1 += 1
            elif value_assigned == 3:
                countb += 1

        o_scalar.InsertNextTuple1(value_assigned)

    omsh.GetCellData().SetScalars(o_scalar)

    countt = countn + count0 + count1 + countb
    count_dic = {
        "total": countt,
        name0: count0,
        name1: count1,
        "overlap": countb,
        "none": countn,
    }

    return omsh, count_dic


# ---------------------------------------------------------------------------
# Tagging functions
# ---------------------------------------------------------------------------


def tag_elements_by_voxel_boxes(
    mesh: vtk.vtkUnstructuredGrid, voxel_bounding_boxes, label_name="scar"
):
    """Tag mesh elements whose centroid falls inside any of the voxel bounding boxes.

    Adds a cell array to the mesh with 1 (inside) or 0 (outside).
    """
    num_cells = mesh.GetNumberOfCells()
    num_bboxes = len(voxel_bounding_boxes)

    scar_array = vtk.vtkIntArray()
    scar_array.SetName(label_name)
    scar_array.SetNumberOfComponents(1)
    scar_array.SetNumberOfTuples(num_cells)

    intersections = np.ndarray((num_bboxes, 2))  # noqa: F841 — verbatim from master
    for cell_id in range(num_cells):
        cell = mesh.GetCell(cell_id)
        num_pts = cell.GetNumberOfPoints()

        centroid = np.zeros(3)
        for i in range(num_pts):
            pt = np.array(mesh.GetPoint(cell.GetPointId(i)))
            centroid += pt
        centroid /= num_pts

        tag = 0
        for jx, box in enumerate(voxel_bounding_boxes):
            if point_in_aabb(centroid, box):
                tag = 1
                break

        scar_array.SetValue(cell_id, tag)

    mesh.GetCellData().AddArray(scar_array)
    return mesh


def tag_mesh_elements_by_voxel_boxes(
    msh, centroids: np.ndarray, voxel_bounding_boxes: list
) -> np.ndarray:
    """Tag mesh elements as '1' if their centroid falls within any voxel bounding box.

    Args:
        centroids (np.ndarray): Nx3 array of mesh element centroids (real-world coords).
        voxel_bounding_boxes (list of np.ndarray): List of 8-corner arrays (8x3) for each voxel.

    Returns:
        np.ndarray: Array of 0/1 tags of shape (N,) for each centroid.
    """
    tags = np.zeros(len(centroids), dtype=np.uint8)

    for box in voxel_bounding_boxes:
        min_corner = np.min(box, axis=0)
        max_corner = np.max(box, axis=0)

        for i, cog in enumerate(centroids):
            if tags[i]:
                continue  # already tagged
            if np.all(cog >= min_corner) and np.all(cog <= max_corner):
                tags[i] = 1

    vtk_array = vtknp.numpy_to_vtk(tags, deep=True, array_type=vtk.VTK_INT)
    vtk_array.SetName("scar")

    msh.GetCellData().AddArray(vtk_array)

    return msh


def tag_mesh_elements_by_growing_from_seed(
    msh, seed_points: np.ndarray, voxel_bounding_boxes: list, cogs=None, label_name="scar"
) -> np.ndarray:
    """seed_points: Nx3 array of seed points in real-world coordinates (centroids of voxel bounding boxes)."""
    num_cells = msh.GetNumberOfCells()
    if cogs is None:
        cogs = get_element_cogs(msh)

    logger.info(f"Building adjacency list for {num_cells} cells.")
    adjacency = _build_adjacency_list(msh)

    tag_array = np.zeros(num_cells, dtype=np.int32)
    visited = np.zeros(num_cells, dtype=bool)

    logger.info("Building cell locator...")
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(msh)
    cell_locator.BuildLocator()

    logger.info(f"Processing {len(seed_points)} seed points...")
    for seed in seed_points:
        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)

        cell_locator.FindClosestPoint(seed, closest_point, cell_id, sub_id, dist2)
        seed_idx = cell_id  # already an integer

        if visited[seed_idx]:
            continue

        queue = deque([seed_idx])
        visited[seed_idx] = True

        while queue:
            current_cell_id = queue.popleft()
            cog = cogs[current_cell_id]
            tag_array[current_cell_id] = 1

            if any(point_in_aabb(cog, box) for box in voxel_bounding_boxes):
                tag_array[current_cell_id] = 1
                for neighbor in adjacency[current_cell_id]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

    tag_vtk_array = vtknp.numpy_to_vtk(tag_array, deep=True, array_type=vtk.VTK_INT)
    tag_vtk_array.SetName(label_name)
    msh.GetCellData().AddArray(tag_vtk_array)

    return msh


def tag_mesh_elements_by_growing_from_seed_optimized(
    msh,
    seed_points: np.ndarray,
    voxel_bounding_boxes: List,
    cogs: Optional[np.ndarray] = None,
    label_name: str = "scar",
    label_value: int = 1,
) -> np.ndarray:
    """Optimized version with pre-filtering and improved BFS.

    Key optimizations:
    1. Pre-identify valid cells within bounding boxes
    2. Stop growing when reaching boundary of valid region
    3. Use sets for faster membership testing
    4. Vectorized bounding box checks

    Tagged cells are set to ``label_value`` in the cell field ``label_name``.
    If that field already exists on the mesh it is overlaid (other cells keep
    their values); otherwise it is created from zeros.
    """
    num_cells = msh.GetNumberOfCells()

    if cogs is None:
        cogs = get_element_cogs(msh)

    # Seed the working array from the existing field (overlay) or zeros.
    tag_array = _seed_cell_field(msh, label_name, num_cells)

    # Pre-compute which cells are within any bounding box
    logger.info("Pre-computing valid cells within bounding boxes...")
    valid_cells = precompute_valid_cells(cogs, voxel_bounding_boxes)
    logger.info(f"Found {len(valid_cells)} valid cells out of {num_cells}")

    if len(valid_cells) == 0:
        logger.warning("No cells found within bounding boxes! Field left unchanged.")
        # Overlay added nothing; write back the (existing or fresh) field as-is.
        _attach_cell_field(msh, tag_array, label_name)
        return msh

    logger.info(f"Building adjacency list for {num_cells} cells.")
    adjacency = _build_adjacency_list_optimized(msh)

    visited = np.zeros(num_cells, dtype=bool)
    num_tagged = 0

    logger.info("Building cell locator...")
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(msh)
    cell_locator.BuildLocator()

    logger.info(f"Processing {len(seed_points)} seed points...")

    for i, seed in enumerate(seed_points):
        if i % 10 == 0:  # Progress logging
            logger.info(f"Processing seed {i + 1}/{len(seed_points)}")

        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)

        cell_locator.FindClosestPoint(seed, closest_point, cell_id, sub_id, dist2)
        seed_idx = cell_id.get()

        if visited[seed_idx] or seed_idx not in valid_cells:
            continue

        # Modified BFS that only grows within valid region
        queue = deque([seed_idx])
        visited[seed_idx] = True

        while queue:
            current_cell_id = queue.popleft()

            # Only tag if cell is in valid region
            if current_cell_id in valid_cells:
                tag_array[current_cell_id] = label_value
                num_tagged += 1

                # Only expand to neighbors if current cell is valid
                for neighbor in adjacency[current_cell_id]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        # Add to queue regardless - we'll check validity when processing
                        queue.append(neighbor)

    # Write the merged field back onto the mesh.
    _attach_cell_field(msh, tag_array, label_name)

    logger.info(f"Tagged {num_tagged} cells with {label_name}={label_value} out of {num_cells}")
    return msh


def tag_mesh_elements_parallel_regions(
    msh,
    seed_points: np.ndarray,
    voxel_bounding_boxes: List,
    cogs: Optional[np.ndarray] = None,
    label_name: str = "scar",
) -> np.ndarray:
    """Alternative approach: Process each bounding box region separately.

    This can be more efficient when bounding boxes are well-separated.
    """
    num_cells = msh.GetNumberOfCells()

    if cogs is None:
        cogs = get_element_cogs(msh)

    logger.info("Building adjacency list...")
    adjacency = _build_adjacency_list_optimized(msh)

    tag_array = np.zeros(num_cells, dtype=np.int32)
    visited = np.zeros(num_cells, dtype=bool)

    logger.info("Building cell locator...")
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(msh)
    cell_locator.BuildLocator()

    # Process each bounding box region separately
    for box_idx, box in enumerate(voxel_bounding_boxes):
        logger.info(f"Processing bounding box {box_idx + 1}/{len(voxel_bounding_boxes)}")

        # Find cells in this bounding box
        box_cells = set()
        for cell_id in range(num_cells):
            if point_in_aabb(cogs[cell_id], [box]):
                box_cells.add(cell_id)

        if not box_cells:
            continue

        # Find seeds relevant to this bounding box
        relevant_seeds = []
        for seed in seed_points:
            closest_point = [0.0, 0.0, 0.0]
            cell_id = vtk.reference(0)
            sub_id = vtk.reference(0)
            dist2 = vtk.reference(0.0)

            cell_locator.FindClosestPoint(seed, closest_point, cell_id, sub_id, dist2)
            seed_idx = cell_id.get()

            if seed_idx in box_cells:
                relevant_seeds.append(seed_idx)

        # Grow from seeds within this bounding box
        for seed_idx in relevant_seeds:
            if visited[seed_idx]:
                continue

            queue = deque([seed_idx])
            visited[seed_idx] = True

            while queue:
                current_cell_id = queue.popleft()

                if current_cell_id in box_cells:
                    tag_array[current_cell_id] = 1

                    for neighbor in adjacency[current_cell_id]:
                        if not visited[neighbor] and neighbor in box_cells:
                            visited[neighbor] = True
                            queue.append(neighbor)

    # Create VTK array and add to mesh
    tag_vtk_array = vtknp.numpy_to_vtk(tag_array, deep=True, array_type=vtk.VTK_INT)
    tag_vtk_array.SetName(label_name)
    msh.GetCellData().AddArray(tag_vtk_array)

    logger.info(f"Tagged {np.sum(tag_array)} cells out of {num_cells}")
    return msh


# ---------------------------------------------------------------------------
# write_vtk re-export (needed by test_compare_mesh_sizes which imports it
# from imatools.core.mesh alongside compare_mesh_sizes)
# ---------------------------------------------------------------------------


def write_vtk(mesh, directory, outname="output", output_type="polydata"):
    """Thin delegator — authoritative implementation lives in vtktools."""
    return _vtk().write_vtk(mesh, directory, outname, output_type)


# ---------------------------------------------------------------------------
# rotate_mesh — migrated from imatools.common.utils (T2b2)
# ---------------------------------------------------------------------------


def rotate_mesh(
    plt_msh,
    lv_tag=1,
    mv_tag=7,
    tv_tag=8,
    fibres=None,
):
    print(
        "Aligning mesh to centre it in 0,0,0 and to have the posterior-anterior direction as 0,-1,0..."
    )

    pts = plt_msh.points
    elem = plt_msh.cells
    elem = np.reshape(elem, (int(plt_msh.cells.shape[0] / 5), 5))
    elem = elem[:, 1:]

    tags = plt_msh.cell_data["ID"]
    eidx_lv = np.where(tags == lv_tag)[0]
    vtx_lv = np.unique(elem[eidx_lv, :].flatten())
    eidx_mv = np.where(tags == mv_tag)[0]
    vtx_mv = np.unique(elem[eidx_mv, :].flatten())
    eidx_tv = np.where(tags == tv_tag)[0]
    vtx_tv = np.unique(elem[eidx_tv, :].flatten())

    cog_mv = np.mean(pts[vtx_mv, :], axis=0)
    cog_tv = np.mean(pts[vtx_tv, :], axis=0)

    dd = np.linalg.norm(pts[vtx_lv, :] - cog_mv, axis=1)
    idx_apex = vtx_lv[np.argmax(dd)]

    cog = np.mean(np.array([cog_mv, cog_tv, pts[idx_apex, :]]), axis=0)

    pts_transformed = plt_msh.points - cog

    v0 = cog_tv - cog_mv
    v0 = v0 / np.linalg.norm(v0)
    v1 = pts[idx_apex, :] - cog_mv
    v1 = v1 / np.linalg.norm(v1)
    n = np.cross(v0, v1)

    n = n / np.linalg.norm(n)

    #### Rotate so the anterior direction is at the front

    target_direction = np.array([0, -1, 0])

    axis_of_rotation = np.cross(n, target_direction)
    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)

    angle = math.acos(np.dot(n, target_direction))
    R = rotation_matrix(axis_of_rotation, angle)  # noqa: N806

    for i in range(pts.shape[0]):
        pts_transformed[i, :] = np.dot(R, pts_transformed[i, :])

    if fibres is not None:
        fibres_transformed = copy.deepcopy(fibres)
        for i in range(fibres.shape[0]):
            fibres_transformed[i, :] = np.dot(R, fibres[i, :])

    # Rotate so the apex is at the bottom
    target_direction_y = np.array([0, 0, -1])

    cog_mv = np.mean(pts_transformed[vtx_mv, :], axis=0)
    long_axis = pts_transformed[idx_apex, :] - cog_mv
    long_axis = long_axis / np.linalg.norm(long_axis)

    angle_y = np.arccos(np.clip(np.dot(long_axis, target_direction_y), -1.0, 1.0))

    cross_product = np.cross(long_axis, target_direction_y)

    ### To take into account clockwise and anticlockwise angles
    if np.linalg.norm(cross_product) != 0:
        direction = np.sign(np.dot(cross_product, np.array([0, -1, 0])))
        angle_y *= direction

    print(f"Long axis: {long_axis}\nTarget direction: {target_direction_y}\nAngle: {angle_y}")
    R_y = rotation_matrix(target_direction, angle_y)  # noqa: N806

    for i in range(pts.shape[0]):
        pts_transformed[i, :] = np.dot(R_y, pts_transformed[i, :])

    if fibres is not None:
        for i in range(fibres.shape[0]):
            fibres_transformed[i, :] = np.dot(R_y, fibres_transformed[i, :])

    plt_msh.points = pts_transformed

    if fibres is not None:
        return plt_msh, fibres_transformed
    else:
        return plt_msh
