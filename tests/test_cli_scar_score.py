"""Characterization tests for the ``imatools-scar score`` subcommand.

New feature (TICKET_scar_score) with NO master oracle, so the expected values
are computed by hand from a synthetic cell-scalar mesh rather than a golden.

Behaviour under test:
  * threshold = ``core.scar.get_threshold(method, value, mean_bp, std_bp)``
  * score = count(cell scalar >= threshold) / count(cell scalar != CEMRGAPP_IGNORE)
  * one ``value | threshold | score`` row per ``--value``.

The synthetic mesh has cell scalars ``[0.3, 0.6, 0.9, 3.0, 0.5]``. With
``CEMRGAPP_IGNORE == 3`` the ``3.0`` cell is dropped from the denominator, so
the eligible total is 4 (values 0.3, 0.6, 0.9, 0.5).
"""

from __future__ import annotations

import vtk

from imatools.cli import scar
from imatools.core.mesh import fibrosis_score
from imatools.core.scar import CEMRGAPP_IGNORE

_CELL_SCALARS = [0.3, 0.6, 0.9, 3.0, 0.5]


def _write_scored_mesh(path, values=_CELL_SCALARS, field="scalars"):
    """Write a 5-cell triangulated polydata with named float cell scalars."""
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    for xyz in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0.5, 0.5, 1)]:
        pts.InsertNextPoint(*xyz)
    pd.SetPoints(pts)

    cells = vtk.vtkCellArray()
    for a, b, c in [(0, 1, 2), (1, 3, 2), (0, 2, 4), (1, 0, 4), (2, 3, 4)]:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, a)
        tri.GetPointIds().SetId(1, b)
        tri.GetPointIds().SetId(2, c)
        cells.InsertNextCell(tri)
    pd.SetPolys(cells)

    arr = vtk.vtkFloatArray()
    arr.SetName(field)
    for v in values:
        arr.InsertNextValue(v)
    pd.GetCellData().SetScalars(arr)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(pd)
    writer.SetFileName(path)
    writer.Write()
    return path


# ---------------------------------------------------------------------------
# Pure core: fibrosis_score with the new exclude_value param
# ---------------------------------------------------------------------------


def _load(path):
    from imatools.io.mesh_io import read_vtk

    return read_vtk(path)


def test_fibrosis_score_exclude_value_default_unchanged(tmp_path):
    """Default exclude_value=0 preserves historical behaviour (no cell == 0 here)."""
    msh = _load(_write_scored_mesh(str(tmp_path / "m.vtk")))
    # th=0.5: eligible total = 5 (nothing == 0); >=0.5 -> 0.6,0.9,3.0,0.5 = 4.
    assert fibrosis_score(msh, 0.5, type="cell") == 4.0 / 5.0


def test_fibrosis_score_exclude_value_three(tmp_path):
    msh = _load(_write_scored_mesh(str(tmp_path / "m.vtk")))
    # th=0.5, exclude 3.0: total = 4; >=0.5 -> 0.6,0.9,0.5 = 3 -> 0.75.
    score = fibrosis_score(msh, 0.5, type="cell", exclude_value=CEMRGAPP_IGNORE)
    assert score == 0.75


# ---------------------------------------------------------------------------
# CLI: imatools-scar score
# ---------------------------------------------------------------------------


def test_score_subcommand_registered():
    parser = scar._build_parser()
    sub = [a for a in parser._actions if a.dest == "command"][0]
    assert "score" in sub.choices


def test_score_iir_single_value(tmp_path, capsys):
    mesh = _write_scored_mesh(str(tmp_path / "m.vtk"))
    # iir: th = value * mean_bp = 0.5 * 1.0 = 0.5 -> score 0.75.
    ret = scar.main(
        [
            "score",
            "--mesh",
            mesh,
            "--method",
            "iir",
            "--value",
            "0.5",
            "--mean-bp",
            "1.0",
            "--stdev-bp",
            "0.1",
        ]
    )
    assert ret == 0
    out = capsys.readouterr().out
    assert "value\tthreshold\tscore" in out
    assert "0.50\t0.50\t0.75000" in out


def test_score_msd_multiple_values(tmp_path, capsys):
    mesh = _write_scored_mesh(str(tmp_path / "m.vtk"))
    # msd: th = value * std + mean. mean=0.0, std=1.0 -> th == value.
    #   value 0.5 -> th 0.5 -> score 0.75 (0.6,0.9,0.5 pass; 3.0 excluded)
    #   value 1.2 -> th 1.2 -> score 0.0  (none of 0.3,0.6,0.9,0.5 >= 1.2)
    ret = scar.main(
        [
            "score",
            "--mesh",
            mesh,
            "--method",
            "msd",
            "--value",
            "0.5",
            "1.2",
            "--mean-bp",
            "0.0",
            "--stdev-bp",
            "1.0",
        ]
    )
    assert ret == 0
    out = capsys.readouterr().out
    assert "0.50\t0.50\t0.75000" in out
    assert "1.20\t1.20\t0.00000" in out


def test_score_named_field(tmp_path, capsys):
    mesh = _write_scored_mesh(str(tmp_path / "m.vtk"), field="lge")
    ret = scar.main(
        [
            "score",
            "--mesh",
            mesh,
            "--method",
            "iir",
            "--value",
            "0.5",
            "--mean-bp",
            "1.0",
            "--stdev-bp",
            "0.1",
            "--field",
            "lge",
        ]
    )
    assert ret == 0
    assert "0.50\t0.50\t0.75000" in capsys.readouterr().out


def test_score_missing_field_errors(tmp_path):
    mesh = _write_scored_mesh(str(tmp_path / "m.vtk"), field="lge")
    ret = scar.main(
        [
            "score",
            "--mesh",
            mesh,
            "--method",
            "iir",
            "--value",
            "0.5",
            "--mean-bp",
            "1.0",
            "--stdev-bp",
            "0.1",
            "--field",
            "does_not_exist",
        ]
    )
    assert ret == 1
