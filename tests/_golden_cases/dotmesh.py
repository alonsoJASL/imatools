"""Capture cases for dotmesh parsers.

Covers:
- ``parse_dotmesh_file`` from ``common/vtktools.py`` — parses a Biosense .mesh file
  into three dicts (general attributes, vertices section, triangles section).
- ``save_array`` from ``convert_dotmesh.py`` — writes a points or elements array to
  a text file; captured via write-then-read (its "output" is the written file content).

Both target ``imatools.parsers.dotmesh`` (T2b4 migration).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import _fixtures as fx
import convert_dotmesh as cdm
from _capture_golden import CaptureCase

from imatools.common import vtktools

# ---------------------------------------------------------------------------
# Inputs shared across cases
# ---------------------------------------------------------------------------

# Write the synthetic .mesh file to a permanent temp directory for the capture run.
# We use a module-level temp dir so all cases in this module share the same file.
_tmp_dir = Path(tempfile.mkdtemp())
_dotmesh_path = fx.write_dotmesh(_tmp_dir, name="synthetic")

# Small int array for save_array (points: shape (N, 3) float)
_pts, _elem, _, _ = fx.carp_mesh()


# ---------------------------------------------------------------------------
# Reduce helpers
# ---------------------------------------------------------------------------


def _reduce_parse_result(result):
    """Map parse_dotmesh_file output to a JSON-serializable dict."""
    gen_attr, verts, tris = result
    return {
        "general_attributes": {
            "MeshID": gen_attr["MeshID"],
            "MeshName": gen_attr["MeshName"],
            "NumVertex": gen_attr["NumVertex"],
            "NumTriangle": gen_attr["NumTriangle"],
        },
        "vertices": {
            "index": verts["index"].tolist(),
            "points": verts["points"].tolist(),
        },
        "triangles": {
            "index": tris["index"].tolist(),
            "elements": tris["elements"].tolist(),
        },
    }


def _save_array_pts():
    """Call save_array with is_elem=False, return written file lines as a list."""
    out = _tmp_dir / "save_array_pts_out.pts"
    cdm.save_array(_pts, str(out), is_elem=False)
    return out.read_text(encoding="utf-8").splitlines()


def _save_array_elem():
    """Call save_array with is_elem=True, return written file lines as a list."""
    out = _tmp_dir / "save_array_elem_out.elem"
    cdm.save_array(_elem, str(out), is_elem=True)
    return out.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

CASES = [
    CaptureCase(
        name="dotmesh/parse_dotmesh_file_synthetic",
        func=vtktools.parse_dotmesh_file,
        args=(str(_dotmesh_path),),
        kwargs={"myencoding": "utf-8"},
        reduce=_reduce_parse_result,
        fmt="json",
    ),
    CaptureCase(
        name="dotmesh/save_array_pts",
        func=_save_array_pts,
        args=(),
        kwargs={},
        reduce=lambda x: x,
        fmt="json",
    ),
    CaptureCase(
        name="dotmesh/save_array_elem",
        func=_save_array_elem,
        args=(),
        kwargs={},
        reduce=lambda x: x,
        fmt="json",
    ),
]
