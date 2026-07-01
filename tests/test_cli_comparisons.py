"""
Tests for the imatools-comparisons CLI (M1.7).

Covers manifest-CSV parsing and a minimal map-fibres wiring path against a
tiny synthetic pair of meshes. No golden fixtures needed — this CLI is pure
orchestration over already golden-backed core/metrics + core/mesh functions.
"""

import pandas as pd
import pytest
import vtk

from imatools.cli import comparisons


def test_comparisons_cli_stub():
    """CLI module loads and exposes main()."""
    assert hasattr(comparisons, "main")


def test_load_manifest_valid(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "comparison_dir,case_left,case_right\n" "/data/C0,caseA,caseB\n" "/data/C1,caseC,caseD\n"
    )
    df = comparisons.load_manifest(str(manifest_path))
    assert list(df.columns) == ["comparison_dir", "case_left", "case_right"]
    assert len(df) == 2
    assert df.iloc[0]["case_left"] == "caseA"


def test_load_manifest_missing_columns(tmp_path):
    manifest_path = tmp_path / "bad_manifest.csv"
    manifest_path.write_text("comparison_dir,case_left\n/data/C0,caseA\n")
    with pytest.raises(ValueError):
        comparisons.load_manifest(str(manifest_path))


def _write_small_triangle_mesh(path, n_triangles):
    """Write a tiny vtkPolyData made of disjoint triangles to a legacy .vtk file.

    core.mesh.create_mapping (via extractPointsAndElemsFromVtk) assumes every
    cell has 3 point ids, so the fixture must use triangle cells rather than
    vertex/point-cloud cells.
    """
    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    for t in range(n_triangles):
        base = 3 * t
        points.InsertNextPoint(float(base), 0.0, 0.0)
        points.InsertNextPoint(float(base) + 1.0, 0.0, 0.0)
        points.InsertNextPoint(float(base), 1.0, 0.0)

        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, base)
        triangle.GetPointIds().SetId(1, base + 1)
        triangle.GetPointIds().SetId(2, base + 2)
        triangles.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.Write()


def test_map_fibres_end_to_end(tmp_path):
    """Exercise handle_map_fibres over a manifest with one comparison of tiny meshes."""
    comparison_dir = tmp_path / "C0"
    case_left = comparison_dir / "left_case"
    case_right = comparison_dir / "right_case"
    case_left.mkdir(parents=True)
    case_right.mkdir(parents=True)

    _write_small_triangle_mesh(case_left / "input.vtk", n_triangles=5)
    _write_small_triangle_mesh(case_right / "input.vtk", n_triangles=3)

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "comparison_dir,case_left,case_right\n" f"{comparison_dir},left_case,right_case\n"
    )

    args = [
        "map-fibres",
        "--manifest",
        str(manifest_path),
        "-n",
        "in",
        "-map",
        "pts",
    ]
    ret = comparisons.main(args)
    assert ret == 0

    out_csv = comparison_dir / "MAPPING" / "input_pts.csv"
    assert out_csv.exists()

    df = pd.read_csv(out_csv)
    assert len(df) > 0
