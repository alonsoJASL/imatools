"""Capture cases: vtktools file I/O surface → ``io/mesh_io``.

Functions characterised (master ``common/vtktools.py``):
  - readVtk          (deprecated wrapper → read_vtk)
  - read_vtk         (polydata and ugrid readers)
  - writeVtk         (deprecated wrapper → write_vtk)
  - write_vtk        (polydata and ugrid writers)
  - clean_stl_file   (strips trailing content after endsolid)
  - export_as        (export to ply/stl/obj/vtp)
  - saveCarpAsVtk    (builds VTK polydata from pts/el arrays)
  - vtk_from_points_file  (builds vtkPolyData from a delimited text file)

File-I/O pattern
~~~~~~~~~~~~~~~~
Writers: call the function with a temp dir/path, then read the written file back
and reduce the content to a stable serializable form.
Readers: write a valid input file to temp, then characterise the read result.

Reduce helpers
~~~~~~~~~~~~~~
Every VTK mesh is reduced to a stable dict so floating-point metadata
differences in writer defaults do not cause false failures:
  ``_reduce_polydata``  → dict with keys n_points, n_cells, points, bounds
  ``_reduce_ugrid``     → dict with keys n_points, n_cells, points, bounds

STL files are reduced to their text content (truncated to the functional lines
before and including endsolid).

All temp directories are created at module load time and kept alive for the
module lifetime (gc'd automatically when the interpreter exits after capture).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import _fixtures as fx
import numpy as np
import vtk
from _capture_golden import CaptureCase

from imatools.common import vtktools

# ---------------------------------------------------------------------------
# Persistent temp directory (module-level; lives for the duration of capture)
# ---------------------------------------------------------------------------

_TMPDIR = tempfile.TemporaryDirectory(prefix="imatools_mesh_io_capture_")
_TMP = Path(_TMPDIR.name)


# ---------------------------------------------------------------------------
# Reduce helpers
# ---------------------------------------------------------------------------


def _reduce_polydata(msh: vtk.vtkPolyData) -> Dict[str, Any]:
    """Reduce a vtkPolyData to a stable JSON-serialisable dict."""
    n_pts = msh.GetNumberOfPoints()
    pts = np.array([msh.GetPoint(i) for i in range(n_pts)], dtype=float).tolist()
    bounds = list(msh.GetBounds())
    return {
        "n_points": n_pts,
        "n_cells": msh.GetNumberOfCells(),
        "points": pts,
        "bounds": bounds,
    }


def _reduce_ugrid(msh: vtk.vtkUnstructuredGrid) -> Dict[str, Any]:
    """Reduce a vtkUnstructuredGrid to a stable JSON-serialisable dict."""
    n_pts = msh.GetNumberOfPoints()
    pts = np.array([msh.GetPoint(i) for i in range(n_pts)], dtype=float).tolist()
    bounds = list(msh.GetBounds())
    return {
        "n_points": n_pts,
        "n_cells": msh.GetNumberOfCells(),
        "points": pts,
        "bounds": bounds,
    }


# ---------------------------------------------------------------------------
# Write fixture files to temp dir (once, at module load)
# ---------------------------------------------------------------------------

# Write a polydata VTK file for read_vtk / readVtk
_PD_VTK_PATH = str(_TMP / "polydata.vtk")
vtktools.write_vtk(fx.polydata(), str(_TMP), "polydata", output_type="polydata")

# Write an unstructured grid VTK file for read_vtk ugrid
_UG_VTK_PATH = str(_TMP / "ugrid.vtk")
vtktools.write_vtk(fx.unstructured_grid(), str(_TMP), "ugrid", output_type="ugrid")

# Write a minimal STL file for clean_stl_file
_STL_DIRTY_PATH = str(_TMP / "dirty.stl")
_STL_DIRTY_CONTENT = (
    "solid test\n"
    "  facet normal 0 0 1\n"
    "    outer loop\n"
    "      vertex 0 0 0\n"
    "      vertex 1 0 0\n"
    "      vertex 0 1 0\n"
    "    endloop\n"
    "  endfacet\n"
    "endsolid test\n"
    "GARBAGE DATA THAT SHOULD BE STRIPPED\n"
    "more garbage\n"
)
with open(_STL_DIRTY_PATH, "w", encoding="utf-8") as _f:
    _f.write(_STL_DIRTY_CONTENT)

# Write a points CSV file for vtk_from_points_file
_POINTS_CSV_PATH = str(_TMP / "points.csv")
_pts_arr = np.array(fx.carp_mesh()[0], dtype=float)
with open(_POINTS_CSV_PATH, "w", encoding="utf-8") as _f:
    for row in _pts_arr:
        _f.write(f"{row[0]},{row[1]},{row[2]}\n")


# ---------------------------------------------------------------------------
# Wrapper functions for writer-type cases (temp-write-then-read)
# ---------------------------------------------------------------------------


def _capture_write_vtk_polydata() -> Dict[str, Any]:
    """write_vtk polydata → read back with vtkPolyDataReader."""
    out_dir = str(_TMP / "write_polydata")
    Path(out_dir).mkdir(exist_ok=True)
    vtktools.write_vtk(fx.polydata(), out_dir, "out_pd", output_type="polydata")
    back = vtktools.read_vtk(str(Path(out_dir) / "out_pd.vtk"), input_type="polydata")
    return _reduce_polydata(back)


def _capture_write_vtk_ugrid() -> Dict[str, Any]:
    """write_vtk ugrid → read back with vtkUnstructuredGridReader."""
    out_dir = str(_TMP / "write_ugrid")
    Path(out_dir).mkdir(exist_ok=True)
    vtktools.write_vtk(fx.unstructured_grid(), out_dir, "out_ug", output_type="ugrid")
    back = vtktools.read_vtk(str(Path(out_dir) / "out_ug.vtk"), input_type="ugrid")
    return _reduce_ugrid(back)


def _capture_write_vtk_deprecated_polydata() -> Dict[str, Any]:
    """writeVtk (deprecated) → read back."""
    out_dir = str(_TMP / "writevtk_pd")
    Path(out_dir).mkdir(exist_ok=True)
    vtktools.writeVtk(fx.polydata(), out_dir, "dep_pd", output_type="polydata")
    back = vtktools.read_vtk(str(Path(out_dir) / "dep_pd.vtk"), input_type="polydata")
    return _reduce_polydata(back)


def _capture_clean_stl_file() -> str:
    """clean_stl_file: strip content after endsolid → return cleaned text."""
    out_path = str(_TMP / "clean.stl")
    vtktools.clean_stl_file(_STL_DIRTY_PATH, out_path)
    with open(out_path, "r", encoding="utf-8") as f:
        return f.read()


def _capture_export_as_ply() -> Dict[str, Any]:
    """export_as ply → read back with vtkPLYReader → reduce."""
    out_path = str(_TMP / "exported.ply")
    vtktools.export_as(fx.polydata(), out_path, export_as="ply")
    reader = vtk.vtkPLYReader()
    reader.SetFileName(out_path)
    reader.Update()
    return _reduce_polydata(reader.GetOutput())


def _capture_export_as_stl() -> Dict[str, Any]:
    """export_as stl → read back with vtkSTLReader → reduce."""
    out_path = str(_TMP / "exported.stl")
    vtktools.export_as(fx.polydata(), out_path, export_as="stl")
    reader = vtk.vtkSTLReader()
    reader.SetFileName(out_path)
    reader.Update()
    return _reduce_polydata(reader.GetOutput())


def _capture_export_as_vtp() -> Dict[str, Any]:
    """export_as vtp → read back with vtkXMLPolyDataReader → reduce."""
    out_path = str(_TMP / "exported.vtp")
    vtktools.export_as(fx.polydata(), out_path, export_as="vtp")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(out_path)
    reader.Update()
    return _reduce_polydata(reader.GetOutput())


def _capture_save_carp_as_vtk() -> Dict[str, Any]:
    """saveCarpAsVtk → write then read back."""
    pts, elem, _region, _lon = fx.carp_mesh()
    out_dir = str(_TMP / "carp_vtk")
    Path(out_dir).mkdir(exist_ok=True)
    vtktools.saveCarpAsVtk(pts, elem, out_dir, "carp_out")
    back = vtktools.read_vtk(str(Path(out_dir) / "carp_out.vtk"), input_type="polydata")
    return _reduce_polydata(back)


def _capture_vtk_from_points_file() -> Dict[str, Any]:
    """vtk_from_points_file → reduce result."""
    result = vtktools.vtk_from_points_file(_POINTS_CSV_PATH, mydelim=",")
    return _reduce_polydata(result)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

CASES = [
    # ------------------------------------------------------------------
    # read_vtk — polydata
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/read_vtk_polydata",
        func=vtktools.read_vtk,
        args=(_PD_VTK_PATH,),
        kwargs={"input_type": "polydata"},
        reduce=_reduce_polydata,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # read_vtk — ugrid
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/read_vtk_ugrid",
        func=vtktools.read_vtk,
        args=(_UG_VTK_PATH,),
        kwargs={"input_type": "ugrid"},
        reduce=_reduce_ugrid,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # readVtk — deprecated wrapper (same result as read_vtk)
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/readVtk_polydata",
        func=vtktools.readVtk,
        args=(_PD_VTK_PATH,),
        kwargs={"input_type": "polydata"},
        reduce=_reduce_polydata,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # write_vtk polydata round-trip
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/write_vtk_polydata",
        func=_capture_write_vtk_polydata,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # write_vtk ugrid round-trip
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/write_vtk_ugrid",
        func=_capture_write_vtk_ugrid,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # writeVtk deprecated wrapper round-trip
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/writeVtk_polydata",
        func=_capture_write_vtk_deprecated_polydata,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # clean_stl_file
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/clean_stl_file",
        func=_capture_clean_stl_file,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # export_as — ply
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/export_as_ply",
        func=_capture_export_as_ply,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # export_as — stl
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/export_as_stl",
        func=_capture_export_as_stl,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # export_as — vtp
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/export_as_vtp",
        func=_capture_export_as_vtp,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # saveCarpAsVtk
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/save_carp_as_vtk",
        func=_capture_save_carp_as_vtk,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # vtk_from_points_file
    # ------------------------------------------------------------------
    CaptureCase(
        name="mesh_io/vtk_from_points_file",
        func=_capture_vtk_from_points_file,
        args=(),
        fmt="json",
    ),
]
