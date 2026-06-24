"""
Foundation self-tests (T0).

These validate the test infrastructure itself — that the synthetic builders produce
valid, deterministic objects and that committed golden values load back through the
``golden`` fixture. They are NOT characterization tests of imatools behaviour (those
arrive in Wave 1); they exist so a broken fixture or golden file fails loudly here
rather than confusingly inside a migration test.
"""

from pathlib import Path

import _fixtures as fx
import numpy as np
import pytest
import SimpleITK as sitk
import vtk
from conftest import GOLDEN_DIR, load_golden

# ---------------------------------------------------------------------------
# Builders produce valid objects
# ---------------------------------------------------------------------------


def test_label_image_geometry(label_image):
    assert isinstance(label_image, sitk.Image)
    assert label_image.GetSize() == (12, 12, 12)
    assert label_image.GetSpacing() == fx.DEFAULT_SPACING
    labels = np.unique(sitk.GetArrayFromImage(label_image))
    assert set(labels.tolist()) == {0, 1, 2, 3}


def test_binary_image_is_binary(binary_image):
    assert set(np.unique(sitk.GetArrayFromImage(binary_image)).tolist()) == {0, 1}


def test_label_image_pair_partial_overlap(label_image_pair):
    a, b = label_image_pair
    arr_a = sitk.GetArrayFromImage(a)
    arr_b = sitk.GetArrayFromImage(b)
    overlap = np.count_nonzero((arr_a > 0) & (arr_b > 0))
    assert 0 < overlap < np.count_nonzero(arr_a > 0)


def test_polydata_shape(polydata):
    assert isinstance(polydata, vtk.vtkPolyData)
    assert polydata.GetNumberOfPoints() == 5
    assert polydata.GetNumberOfCells() == 4
    assert polydata.GetPointData().GetScalars().GetName() == "scalars"
    assert polydata.GetCellData().GetScalars().GetName() == "scalars"


def test_unstructured_grid_shape(unstructured_grid):
    assert isinstance(unstructured_grid, vtk.vtkUnstructuredGrid)
    assert unstructured_grid.GetNumberOfPoints() == 5
    assert unstructured_grid.GetNumberOfCells() == 2


def test_carp_mesh_shapes(carp_mesh):
    pts, elem, region, lon = carp_mesh
    assert pts.shape == (5, 3)
    assert elem.shape == (4, 3)
    assert region.shape == (4,)
    assert lon.shape == (4, 3)


def test_carp_files_written(carp_mesh_files):
    base = Path(str(carp_mesh_files))
    for ext in (".pts", ".elem", ".lon"):
        path = base.with_suffix(ext)
        assert path.exists() and path.stat().st_size > 0
    # CARP convention: first line is the record count.
    assert base.with_suffix(".pts").read_text().splitlines()[0] == "5"
    assert base.with_suffix(".elem").read_text().splitlines()[0] == "4"


def test_dotmesh_written(dotmesh_file):
    text = Path(str(dotmesh_file)).read_text()
    assert "[GeneralAttributes]" in text
    assert "[VerticesSection]" in text
    assert "[TrianglesSection]" in text
    assert "NumVertex = 5" in text
    assert "NumTriangle = 4" in text


# ---------------------------------------------------------------------------
# Determinism — the property the golden contract relies on
# ---------------------------------------------------------------------------


def test_scalar_field_deterministic():
    np.testing.assert_array_equal(fx.scalar_field(), fx.scalar_field())


def test_label_array_deterministic():
    np.testing.assert_array_equal(fx.label_array(), fx.label_array())


# ---------------------------------------------------------------------------
# Golden round-trip — proves capture output loads back through the fixture
# ---------------------------------------------------------------------------


def _golden_stems():
    if not GOLDEN_DIR.exists():
        return []
    stems = sorted(
        str(p.relative_to(GOLDEN_DIR).with_suffix(""))
        for p in GOLDEN_DIR.rglob("*")
        if p.suffix in {".npy", ".json"}
    )
    return stems


@pytest.mark.parametrize("stem", _golden_stems())
def test_golden_values_load(stem):
    value = load_golden(stem)
    assert value is not None


def test_some_golden_values_present():
    """Sanity: the capture harness has been run at least once."""
    if not _golden_stems():
        pytest.skip("no golden values captured yet — run tests/_capture_golden.py against master")
