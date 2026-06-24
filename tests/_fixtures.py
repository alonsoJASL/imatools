"""
Synthetic fixture builders for the imatools characterization test net.

These are **pure, deterministic builders**: same call -> same object, every time.
They depend on numpy / SimpleITK / vtk ONLY and never import ``imatools`` (or any
``common`` module), so the very same builders import cleanly under BOTH layouts:

  * the ``imatools-dev`` env, where pytest fixtures (``conftest.py``) wrap them, and
  * the ``imatools`` (master) env, where the golden-capture harness feeds them to
    master's functions to record the behaviour contract.

That shared provenance is what makes the golden values meaningful: the bytes fed to
master at capture time are bit-for-bit the bytes fed to the migrated code at test time.

Keep every builder deterministic. No ``time``, no unseeded RNG, no filesystem reads.
File-writing helpers (CARP / .mesh) are provided for round-trip I/O characterization;
they take an explicit destination and write nothing implicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk
import vtk

# ---------------------------------------------------------------------------
# SimpleITK images
# ---------------------------------------------------------------------------

#: Default voxel geometry used by every image builder, so spacing/origin-aware
#: functions (volumes, regionprops, resampling) see a non-trivial, fixed header.
DEFAULT_SPACING: Tuple[float, float, float] = (1.0, 1.0, 2.0)
DEFAULT_ORIGIN: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def label_array(shape: Tuple[int, int, int] = (12, 12, 12)) -> np.ndarray:
    """Deterministic multi-label voxel array (background 0, labels 1/2/3).

    Returns a numpy array in (z, y, x) order — i.e. the order
    ``sitk.GetArrayFromImage`` yields — so callers reasoning in either index
    convention have a single source of truth.
    """
    arr = np.zeros(shape, dtype=np.uint8)
    arr[2:6, 2:6, 2:6] = 1
    arr[6:10, 6:10, 6:10] = 2
    arr[3:5, 7:9, 3:5] = 3
    return arr


def label_image(
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
    origin: Tuple[float, float, float] = DEFAULT_ORIGIN,
) -> sitk.Image:
    """Small labeled ``sitk.Image`` (uint8) with three foreground labels."""
    img = sitk.GetImageFromArray(label_array())
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


def binary_image(
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
    origin: Tuple[float, float, float] = DEFAULT_ORIGIN,
) -> sitk.Image:
    """Small binary ``sitk.Image`` (any non-zero label collapsed to 1)."""
    arr = (label_array() > 0).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


def label_image_pair() -> Tuple[sitk.Image, sitk.Image]:
    """Two overlapping labeled images for comparison metrics (dice, etc.).

    The second is the first shifted by one voxel along each axis, giving a
    partial-overlap pair with a stable, known intersection.
    """
    base = label_array()
    shifted = np.zeros_like(base)
    shifted[1:, 1:, 1:] = base[:-1, :-1, :-1]
    img0, img1 = sitk.GetImageFromArray(base), sitk.GetImageFromArray(shifted)
    for img in (img0, img1):
        img.SetSpacing(DEFAULT_SPACING)
        img.SetOrigin(DEFAULT_ORIGIN)
    return img0, img1


# ---------------------------------------------------------------------------
# VTK meshes
# ---------------------------------------------------------------------------

#: Five points forming a small closed-ish surface (square base + apex).
_SURFACE_POINTS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.5, 0.5, 1.0),
)
_SURFACE_TRIANGLES = ((0, 1, 2), (1, 3, 2), (0, 2, 4), (1, 0, 4))

#: A two-tetrahedron volume mesh.
_VOLUME_POINTS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
)
_TETRA_CELLS = ((0, 1, 2, 3), (1, 2, 3, 4))


def _vtk_points(coords) -> vtk.vtkPoints:
    pts = vtk.vtkPoints()
    for x, y, z in coords:
        pts.InsertNextPoint(x, y, z)
    return pts


def _add_scalar_arrays(mesh, n_points: int, n_cells: int) -> None:
    """Attach deterministic point- and cell-data arrays named ``scalars``.

    Many master mesh functions default to a ``scalars`` field; giving both
    associations a known, monotone ramp makes thresholding/scoring testable.
    """
    point_scalars = vtk.vtkFloatArray()
    point_scalars.SetName("scalars")
    for i in range(n_points):
        point_scalars.InsertNextValue(float(i) / max(n_points - 1, 1))
    mesh.GetPointData().SetScalars(point_scalars)

    cell_scalars = vtk.vtkFloatArray()
    cell_scalars.SetName("scalars")
    for i in range(n_cells):
        cell_scalars.InsertNextValue(float(i) + 1.0)
    mesh.GetCellData().SetScalars(cell_scalars)


def polydata() -> vtk.vtkPolyData:
    """Small triangulated ``vtkPolyData`` surface with point/cell ``scalars``."""
    pd = vtk.vtkPolyData()
    pd.SetPoints(_vtk_points(_SURFACE_POINTS))

    cells = vtk.vtkCellArray()
    for a, b, c in _SURFACE_TRIANGLES:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, a)
        tri.GetPointIds().SetId(1, b)
        tri.GetPointIds().SetId(2, c)
        cells.InsertNextCell(tri)
    pd.SetPolys(cells)

    _add_scalar_arrays(pd, len(_SURFACE_POINTS), len(_SURFACE_TRIANGLES))
    return pd


def unstructured_grid() -> vtk.vtkUnstructuredGrid:
    """Small tetrahedral ``vtkUnstructuredGrid`` with point/cell ``scalars``."""
    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(_vtk_points(_VOLUME_POINTS))
    for cell in _TETRA_CELLS:
        tetra = vtk.vtkTetra()
        for i, pid in enumerate(cell):
            tetra.GetPointIds().SetId(i, pid)
        ug.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())

    _add_scalar_arrays(ug, len(_VOLUME_POINTS), len(_TETRA_CELLS))
    return ug


# ---------------------------------------------------------------------------
# CARP text meshes (pts / elem / lon)
# ---------------------------------------------------------------------------


def carp_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """In-memory CARP triangle mesh: ``(pts, elem, region, lon)``.

    * ``pts``    -> (N, 3) float node coordinates
    * ``elem``   -> (M, 3) int triangle connectivity (0-based)
    * ``region`` -> (M,)   int element tags
    * ``lon``    -> (M, 3) float fibre directions (one per element)
    """
    pts = np.array(_SURFACE_POINTS, dtype=float)
    elem = np.array(_SURFACE_TRIANGLES, dtype=int)
    region = np.array([1, 1, 2, 2], dtype=int)
    lon = np.array(
        [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
        dtype=float,
    )
    return pts, elem, region, lon


def write_carp_mesh(directory, name: str = "mesh") -> Path:
    """Write the synthetic CARP mesh as ``<name>.pts/.elem/.lon`` under *directory*.

    Files follow the CARP convention master's readers expect: a single
    count header line, then one row per record. Triangle elements are written
    in ``Tr n0 n1 n2 <tag>`` form; the ``.lon`` file is headed by its fibre count.
    Returns the base path (``directory/name``) the readers key off.
    """
    directory = Path(directory)
    pts, elem, region, lon = carp_mesh()
    base = directory / name

    with open(f"{base}.pts", "w", encoding="utf-8") as fh:
        fh.write(f"{len(pts)}\n")
        for x, y, z in pts:
            fh.write(f"{x:.12f} {y:.12f} {z:.12f}\n")

    with open(f"{base}.elem", "w", encoding="utf-8") as fh:
        fh.write(f"{len(elem)}\n")
        for (a, b, c), tag in zip(elem, region):
            fh.write(f"Tr {a} {b} {c} {tag}\n")

    with open(f"{base}.lon", "w", encoding="utf-8") as fh:
        fh.write("1\n")
        for fx, fy, fz in lon:
            fh.write(f"{fx:.12f} {fy:.12f} {fz:.12f}\n")

    return base


# ---------------------------------------------------------------------------
# Biosense Webster ".mesh" (dotmesh) text
# ---------------------------------------------------------------------------


def dotmesh_text(name: str = "synthetic") -> str:
    """Return a minimal but valid Biosense ``.mesh`` file body.

    Shaped for ``vtktools.parse_dotmesh_file``: a ``[GeneralAttributes]`` block
    declaring vertex/triangle counts, then ``[VerticesSection]`` and
    ``[TrianglesSection]`` whose ``index = data`` rows carry (at least) three
    leading values the parser slices off.
    """
    verts = _SURFACE_POINTS
    tris = _SURFACE_TRIANGLES
    lines = [
        "[GeneralAttributes]",
        "MeshID = 1",
        f"MeshName = {name}",
        f"NumVertex = {len(verts)}",
        f"NumTriangle = {len(tris)}",
        "",
        "[VerticesSection]",
    ]
    for i, (x, y, z) in enumerate(verts):
        lines.append(f"{i} = {x:.6f} {y:.6f} {z:.6f} 0.000000 0.000000 0.000000")
    lines += ["", "[TrianglesSection]"]
    for i, (a, b, c) in enumerate(tris):
        lines.append(f"{i} = {a} {b} {c} 0 0 0")
    return "\n".join(lines) + "\n"


def write_dotmesh(directory, name: str = "synthetic") -> Path:
    """Write the synthetic ``.mesh`` to ``<directory>/<name>.mesh``; return its path."""
    path = Path(directory) / f"{name}.mesh"
    path.write_text(dotmesh_text(name=name), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Plain numpy arrays (metrics / array math)
# ---------------------------------------------------------------------------


def scalar_field(n: int = 16) -> np.ndarray:
    """Deterministic 1-D float field for scalar-comparison / boxplot metrics."""
    return np.linspace(0.0, 1.0, n, dtype=float)


def vector_field(n: int = 16) -> np.ndarray:
    """Deterministic (n, 3) float field for vector-comparison / l2_norm."""
    base = np.linspace(0.0, 1.0, n, dtype=float)
    return np.stack([base, base[::-1], np.zeros_like(base)], axis=1)


def classification_thresholds():
    """``(low, high)`` half-open bins used by ``classify_array``/``count_values_in_ranges``."""
    return [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
