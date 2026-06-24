"""
Test configuration and fixtures for imatools.

The synthetic fixtures below wrap the pure builders in ``tests/_fixtures.py`` — the
SAME builders the golden-capture harness feeds to master — so characterization tests
exercise the migrated code on byte-identical inputs. ``GOLDEN_DIR`` points at the
committed contract values; ``load_golden`` reads them back.
"""

import json
import shutil
import tempfile
from pathlib import Path

import _fixtures as fx
import numpy as np
import pytest

#: Directory holding the committed golden-master values (see tests/golden/README.md).
GOLDEN_DIR = Path(__file__).parent / "golden"


def load_golden(name: str):
    """Load a committed golden value by stem (``"metrics/foo"``), inferring the format.

    Resolves ``.npy`` first (arrays), then ``.json`` (dicts/scalars/lists). Raises
    ``FileNotFoundError`` if neither exists — usually meaning the capture harness has
    not been run for that case yet.
    """
    npy = GOLDEN_DIR / f"{name}.npy"
    if npy.exists():
        return np.load(npy)
    js = GOLDEN_DIR / f"{name}.json"
    if js.exists():
        return json.loads(js.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no golden file for {name!r} under {GOLDEN_DIR}")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_data_dir():
    """Path to sample test data (to be populated)."""
    data_dir = Path(__file__).parent / "fixtures"
    return data_dir


@pytest.fixture
def golden():
    """Expose :func:`load_golden` as a fixture for characterization tests."""
    return load_golden


# ---------------------------------------------------------------------------
# Synthetic data fixtures (wrap tests/_fixtures.py builders)
# ---------------------------------------------------------------------------


@pytest.fixture
def label_image():
    """Small labeled ``sitk.Image`` (uint8, labels 1/2/3)."""
    return fx.label_image()


@pytest.fixture
def binary_image():
    """Small binary ``sitk.Image``."""
    return fx.binary_image()


@pytest.fixture
def label_image_pair():
    """Two partially overlapping labeled images for comparison metrics."""
    return fx.label_image_pair()


@pytest.fixture
def polydata():
    """Small triangulated ``vtkPolyData`` with point/cell ``scalars``."""
    return fx.polydata()


@pytest.fixture
def unstructured_grid():
    """Small tetrahedral ``vtkUnstructuredGrid`` with point/cell ``scalars``."""
    return fx.unstructured_grid()


@pytest.fixture
def carp_mesh():
    """In-memory CARP triangle mesh ``(pts, elem, region, lon)``."""
    return fx.carp_mesh()


@pytest.fixture
def carp_mesh_files(temp_dir):
    """Synthetic CARP mesh written to disk; yields the base path (no extension)."""
    return fx.write_carp_mesh(temp_dir)


@pytest.fixture
def dotmesh_file(temp_dir):
    """Synthetic Biosense ``.mesh`` written to disk; yields its path."""
    return fx.write_dotmesh(temp_dir)
