"""Characterization tests for ``imatools.io.carp_io`` (T1j).

All tests import from the TARGET location ``imatools.io.carp_io``.

Golden values were captured from master via::

    ~/opt/anaconda3/bin/conda run -n imatools env \\
        PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module carp --out tests/golden

where ``M = ~/dev/python/imatools.worktrees/master``.

``readParseElem`` / ``loadCarpMesh`` were formerly xfail intent tests (master
broke on triangle meshes by hardcoding ``el_type='Tt'``); M3-C1 fixed the loader
to detect the element type, so they are now regular passing tests.
"""

from __future__ import annotations

import os
import tempfile

import _fixtures as fx
import numpy as np

# ---------------------------------------------------------------------------
# read_pts
# ---------------------------------------------------------------------------


def test_read_pts(golden, carp_mesh_files):
    from imatools.io.carp_io import read_pts

    result = read_pts(str(carp_mesh_files) + ".pts")
    expected = golden("carp/read_pts")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# read_elem
# ---------------------------------------------------------------------------


def test_read_elem_tr_notags(golden, carp_mesh_files):
    from imatools.io.carp_io import read_elem

    result = read_elem(str(carp_mesh_files) + ".elem", el_type="Tr", tags=False)
    expected = golden("carp/read_elem_tr_notags")
    np.testing.assert_array_equal(result, expected)


def test_read_elem_tr_tags(golden, carp_mesh_files):
    from imatools.io.carp_io import read_elem

    result = read_elem(str(carp_mesh_files) + ".elem", el_type="Tr", tags=True)
    expected = golden("carp/read_elem_tr_tags")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# read_lon
# ---------------------------------------------------------------------------


def test_read_lon(golden, carp_mesh_files):
    from imatools.io.carp_io import read_lon

    result = read_lon(str(carp_mesh_files) + ".lon")
    expected = golden("carp/read_lon")
    np.testing.assert_allclose(result, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# saveToCarpTxt — captured via read-back of written text
# ---------------------------------------------------------------------------


def test_save_to_carp_txt_pts_lines(golden):
    from imatools.io.carp_io import saveToCarpTxt

    pts, elem, _region, _lon = fx.carp_mesh()
    out_dir = tempfile.mkdtemp()
    mshname = os.path.join(out_dir, "out")
    saveToCarpTxt(pts, elem, mshname)
    with open(mshname + ".pts", encoding="utf-8") as fh:
        result = [line.rstrip("\n") for line in fh.readlines()]
    expected = golden("carp/saveToCarpTxt_pts_lines")
    assert result == expected


def test_save_to_carp_txt_elem_lines(golden):
    from imatools.io.carp_io import saveToCarpTxt

    pts, elem, _region, _lon = fx.carp_mesh()
    out_dir = tempfile.mkdtemp()
    mshname = os.path.join(out_dir, "out")
    saveToCarpTxt(pts, elem, mshname)
    with open(mshname + ".elem", encoding="utf-8") as fh:
        result = [line.rstrip("\n") for line in fh.readlines()]
    expected = golden("carp/saveToCarpTxt_elem_lines")
    assert result == expected


# ---------------------------------------------------------------------------
# readParsePts
# ---------------------------------------------------------------------------


def test_read_parse_pts_nodes(golden, carp_mesh_files):
    from imatools.io.carp_io import readParsePts

    nodes, _num_nodes = readParsePts(str(carp_mesh_files) + ".pts")
    expected = golden("carp/readParsePts_nodes")
    np.testing.assert_allclose(nodes, expected, rtol=1e-7)


def test_read_parse_pts_count(golden, carp_mesh_files):
    from imatools.io.carp_io import readParsePts

    _nodes, num_nodes = readParsePts(str(carp_mesh_files) + ".pts")
    expected = golden("carp/readParsePts_count")
    assert num_nodes == expected


# ---------------------------------------------------------------------------
# readParseElem — triangle-mesh loading (M3-C1: was xfail; master hardcoded
# el_type='Tt', which broke triangle .elem files — now type is detected).
# ---------------------------------------------------------------------------


def test_read_parse_elem_intent(carp_mesh_files):
    from imatools.io.carp_io import readParseElem

    el, n_elem = readParseElem(str(carp_mesh_files) + ".elem")
    assert n_elem == 4
    assert len(el) == n_elem


# ---------------------------------------------------------------------------
# loadCarpMesh — triangle-mesh loading (M3-C1: was xfail; same root cause).
# ---------------------------------------------------------------------------


def test_load_carp_mesh_intent(carp_mesh_files):
    from imatools.io.carp_io import loadCarpMesh

    pts, elem, region = loadCarpMesh(str(carp_mesh_files))
    assert pts.shape == (5, 3)
    assert len(elem) == 4
    assert len(region) == 4
