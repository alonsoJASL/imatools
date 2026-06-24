"""Characterization tests for ``imatools.io.mesh_io`` (T1i).

All tests import from the TARGET location ``imatools.io.mesh_io``.  The dev
module already has 3 functions (``load_mesh``, ``save_mesh``, ``clean_stl_file``)
with different signatures from master.  The full set of master functions will be
reconciled by migration task T2b3.  Until then every test is marked
``xfail(strict=False)`` — functions whose dev version already matches master
behaviour (especially ``clean_stl_file``) will XPASS harmlessly; functions that
differ or are absent will xfail.

Functions characterised (master ``common/vtktools.py`` → ``io/mesh_io``):
  - read_vtk         (polydata and ugrid readers)
  - readVtk          (deprecated wrapper → read_vtk)
  - write_vtk        (polydata and ugrid writers)
  - writeVtk         (deprecated wrapper → write_vtk)
  - clean_stl_file   (strips trailing content after endsolid)
  - export_as        (export to ply/stl/vtp formats)
  - saveCarpAsVtk    (builds VTK polydata from pts/el arrays)
  - vtk_from_points_file  (builds vtkPolyData from a delimited text file)

Golden values were captured from master via::

    M=~/dev/python/imatools.worktrees/master
    ~/opt/anaconda3/bin/conda run -n imatools env \\
        PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module mesh_io --out tests/golden

Comparison notes
----------------
* All goldens are **json** dicts with keys ``n_points``, ``n_cells``, ``points``,
  ``bounds``; or plain string for ``clean_stl_file``.
* Point and bounds comparisons use ``np.testing.assert_allclose`` (float geometry).
* Integer counts use plain ``==``.
* File I/O tests create their own ``tempfile.TemporaryDirectory()`` — independent
  of the capture-time temp paths.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import _fixtures as fx
import numpy as np
import pytest
import vtk

# ---------------------------------------------------------------------------
# Reduce helpers (identical to those in the capture cases module)
# ---------------------------------------------------------------------------


def _reduce_polydata(msh: vtk.vtkPolyData) -> Dict[str, Any]:
    """Reduce a vtkPolyData to a stable dict matching the golden format."""
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
    """Reduce a vtkUnstructuredGrid to a stable dict matching the golden format."""
    n_pts = msh.GetNumberOfPoints()
    pts = np.array([msh.GetPoint(i) for i in range(n_pts)], dtype=float).tolist()
    bounds = list(msh.GetBounds())
    return {
        "n_points": n_pts,
        "n_cells": msh.GetNumberOfCells(),
        "points": pts,
        "bounds": bounds,
    }


def _assert_mesh_dict_equal(result: Dict[str, Any], expected: Dict[str, Any]) -> None:
    """Assert a reduced mesh dict matches the golden.

    * ``n_points``, ``n_cells`` (int) → exact equality
    * ``points``, ``bounds`` (list of float) → allclose
    """
    assert result["n_points"] == expected["n_points"], "n_points mismatch"
    assert result["n_cells"] == expected["n_cells"], "n_cells mismatch"
    np.testing.assert_allclose(
        np.asarray(result["points"]),
        np.asarray(expected["points"]),
        rtol=1e-5,
        err_msg="points mismatch",
    )
    np.testing.assert_allclose(
        np.asarray(result["bounds"]),
        np.asarray(expected["bounds"]),
        rtol=1e-5,
        err_msg="bounds mismatch",
    )


# ---------------------------------------------------------------------------
# File-writing helpers (write fixture inputs to temp dirs at test time)
# ---------------------------------------------------------------------------


def _write_polydata_vtk(tmpdir: Path) -> str:
    """Write the synthetic polydata fixture as a VTK file; return its path."""
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(fx.polydata())
    path = str(tmpdir / "polydata.vtk")
    writer.SetFileName(path)
    writer.SetFileTypeToASCII()
    writer.Update()
    return path


def _write_ugrid_vtk(tmpdir: Path) -> str:
    """Write the synthetic unstructured_grid fixture as a VTK file; return its path."""
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetInputData(fx.unstructured_grid())
    path = str(tmpdir / "ugrid.vtk")
    writer.SetFileName(path)
    writer.SetFileTypeToASCII()
    writer.Update()
    return path


def _write_dirty_stl(tmpdir: Path) -> str:
    """Write an STL file with garbage trailing content; return its path."""
    path = str(tmpdir / "dirty.stl")
    content = (
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
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _write_points_csv(tmpdir: Path) -> str:
    """Write the CARP mesh pts as a CSV file; return its path."""
    pts_arr = np.array(fx.carp_mesh()[0], dtype=float)
    path = str(tmpdir / "points.csv")
    with open(path, "w", encoding="utf-8") as f:
        for row in pts_arr:
            f.write(f"{row[0]},{row[1]},{row[2]}\n")
    return path


# ---------------------------------------------------------------------------
# read_vtk — polydata
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_read_vtk_polydata(golden):
    from imatools.io.mesh_io import read_vtk

    with tempfile.TemporaryDirectory() as tmp:
        vtk_path = _write_polydata_vtk(Path(tmp))
        result = read_vtk(vtk_path, input_type="polydata")
        # Accept either a vtkPolyData or a MeshContract
        if not isinstance(result, vtk.vtkPolyData):
            result = result.get_polydata()
        reduced = _reduce_polydata(result)
        expected = golden("mesh_io/read_vtk_polydata")
        _assert_mesh_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# read_vtk — ugrid
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_read_vtk_ugrid(golden):
    from imatools.io.mesh_io import read_vtk

    with tempfile.TemporaryDirectory() as tmp:
        vtk_path = _write_ugrid_vtk(Path(tmp))
        result = read_vtk(vtk_path, input_type="ugrid")
        # Accept either a vtkUnstructuredGrid or a MeshContract
        if not isinstance(result, vtk.vtkUnstructuredGrid):
            result = result.get_ugrid()
        reduced = _reduce_ugrid(result)
        expected = golden("mesh_io/read_vtk_ugrid")
        _assert_mesh_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# readVtk — deprecated wrapper
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_read_vtk_deprecated_polydata(golden):
    """readVtk is the deprecated wrapper around read_vtk."""
    from imatools.io.mesh_io import readVtk

    with tempfile.TemporaryDirectory() as tmp:
        vtk_path = _write_polydata_vtk(Path(tmp))
        result = readVtk(vtk_path, input_type="polydata")
        if not isinstance(result, vtk.vtkPolyData):
            result = result.get_polydata()
        reduced = _reduce_polydata(result)
        expected = golden("mesh_io/readVtk_polydata")
        _assert_mesh_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# write_vtk — polydata round-trip
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_write_vtk_polydata(golden):
    from imatools.io.mesh_io import read_vtk, write_vtk

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        write_vtk(fx.polydata(), str(tmp_path), "out_pd", output_type="polydata")
        back = read_vtk(str(tmp_path / "out_pd.vtk"), input_type="polydata")
        if not isinstance(back, vtk.vtkPolyData):
            back = back.get_polydata()
        reduced = _reduce_polydata(back)
        expected = golden("mesh_io/write_vtk_polydata")
        _assert_mesh_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# write_vtk — ugrid round-trip
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_write_vtk_ugrid(golden):
    from imatools.io.mesh_io import read_vtk, write_vtk

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        write_vtk(fx.unstructured_grid(), str(tmp_path), "out_ug", output_type="ugrid")
        back = read_vtk(str(tmp_path / "out_ug.vtk"), input_type="ugrid")
        if not isinstance(back, vtk.vtkUnstructuredGrid):
            back = back.get_ugrid()
        reduced = _reduce_ugrid(back)
        expected = golden("mesh_io/write_vtk_ugrid")
        _assert_mesh_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# writeVtk — deprecated wrapper round-trip
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_write_vtk_deprecated_polydata(golden):
    """writeVtk is the deprecated wrapper around write_vtk."""
    from imatools.io.mesh_io import read_vtk, writeVtk

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        writeVtk(fx.polydata(), str(tmp_path), "dep_pd", output_type="polydata")
        back = read_vtk(str(tmp_path / "dep_pd.vtk"), input_type="polydata")
        if not isinstance(back, vtk.vtkPolyData):
            back = back.get_polydata()
        reduced = _reduce_polydata(back)
        expected = golden("mesh_io/writeVtk_polydata")
        _assert_mesh_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# clean_stl_file
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_clean_stl_file(golden):
    from imatools.io.mesh_io import clean_stl_file

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        in_path = _write_dirty_stl(tmp_path)
        out_path = str(tmp_path / "clean.stl")
        clean_stl_file(in_path, out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            result = f.read()
        expected = golden("mesh_io/clean_stl_file")
        assert (
            result == expected
        ), f"cleaned STL content mismatch:\ngot: {result!r}\nexpected: {expected!r}"


# ---------------------------------------------------------------------------
# export_as — ply
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_export_as_ply(golden):
    from imatools.io.mesh_io import export_as

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "exported.ply")
        export_as(fx.polydata(), out_path, export_as="ply")
        reader = vtk.vtkPLYReader()
        reader.SetFileName(out_path)
        reader.Update()
        reduced = _reduce_polydata(reader.GetOutput())
        expected = golden("mesh_io/export_as_ply")
        _assert_mesh_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# export_as — stl
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_export_as_stl(golden):
    from imatools.io.mesh_io import export_as

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "exported.stl")
        export_as(fx.polydata(), out_path, export_as="stl")
        reader = vtk.vtkSTLReader()
        reader.SetFileName(out_path)
        reader.Update()
        reduced = _reduce_polydata(reader.GetOutput())
        expected = golden("mesh_io/export_as_stl")
        _assert_mesh_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# export_as — vtp
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_export_as_vtp(golden):
    from imatools.io.mesh_io import export_as

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "exported.vtp")
        export_as(fx.polydata(), out_path, export_as="vtp")
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(out_path)
        reader.Update()
        reduced = _reduce_polydata(reader.GetOutput())
        expected = golden("mesh_io/export_as_vtp")
        _assert_mesh_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# saveCarpAsVtk
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_save_carp_as_vtk(golden):
    from imatools.io.mesh_io import read_vtk, saveCarpAsVtk

    pts, elem, _region, _lon = fx.carp_mesh()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        saveCarpAsVtk(pts, elem, str(tmp_path), "carp_out")
        back = read_vtk(str(tmp_path / "carp_out.vtk"), input_type="polydata")
        if not isinstance(back, vtk.vtkPolyData):
            back = back.get_polydata()
        reduced = _reduce_polydata(back)
        expected = golden("mesh_io/save_carp_as_vtk")
        _assert_mesh_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# vtk_from_points_file
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2b3", strict=False)
def test_vtk_from_points_file(golden):
    from imatools.io.mesh_io import vtk_from_points_file

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_points_csv(Path(tmp))
        result = vtk_from_points_file(csv_path, mydelim=",")
        if not isinstance(result, vtk.vtkPolyData):
            result = result.get_polydata()
        reduced = _reduce_polydata(result)
        expected = golden("mesh_io/vtk_from_points_file")
        _assert_mesh_dict_equal(reduced, expected)
