"""Characterization tests for imatools.parsers.dotmesh.

These tests pin the behaviour of:
- ``parse_dotmesh_file`` — parse a Biosense .mesh file into three structured dicts.
- ``save_array``          — write a points or elements array to a CARP-style text file.

All tests are marked xfail (strict=False) because ``imatools.parsers.dotmesh`` is an
empty stub pending migration task T2b4. Remove the xfail decorators once migration lands.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import _fixtures as fx
import numpy as np
import pytest  # noqa: F401 (used via pytest.mark)

# ---------------------------------------------------------------------------
# Helpers (mirror the reduce logic from _golden_cases/dotmesh.py)
# ---------------------------------------------------------------------------


def _reduce_parse_result(result):
    """Map parse_dotmesh_file output to a JSON-comparable dict."""
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


# ---------------------------------------------------------------------------
# parse_dotmesh_file
# ---------------------------------------------------------------------------


def test_parse_dotmesh_file_synthetic(golden):
    from imatools.parsers.dotmesh import parse_dotmesh_file

    with tempfile.TemporaryDirectory() as tmp:
        mesh_path = fx.write_dotmesh(tmp, name="synthetic")
        result = parse_dotmesh_file(str(mesh_path), myencoding="utf-8")

    actual = _reduce_parse_result(result)
    expected = golden("dotmesh/parse_dotmesh_file_synthetic")

    assert actual["general_attributes"] == expected["general_attributes"]
    assert actual["vertices"]["index"] == expected["vertices"]["index"]
    assert actual["triangles"]["index"] == expected["triangles"]["index"]
    assert np.allclose(actual["vertices"]["points"], expected["vertices"]["points"])
    assert actual["triangles"]["elements"] == expected["triangles"]["elements"]


# ---------------------------------------------------------------------------
# save_array — is_elem=False (points)
# ---------------------------------------------------------------------------


def test_save_array_pts(golden):
    from imatools.parsers.dotmesh import save_array

    pts, _, _, _ = fx.carp_mesh()

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "out.pts"
        save_array(pts, str(out_path), is_elem=False)
        actual_lines = out_path.read_text(encoding="utf-8").splitlines()

    expected_lines = golden("dotmesh/save_array_pts")
    assert actual_lines == expected_lines


# ---------------------------------------------------------------------------
# save_array — is_elem=True (elements)
# ---------------------------------------------------------------------------


def test_save_array_elem(golden):
    from imatools.parsers.dotmesh import save_array

    _, elem, _, _ = fx.carp_mesh()

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "out.elem"
        save_array(elem, str(out_path), is_elem=True)
        actual_lines = out_path.read_text(encoding="utf-8").splitlines()

    expected_lines = golden("dotmesh/save_array_elem")
    assert actual_lines == expected_lines
