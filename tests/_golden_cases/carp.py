"""Capture cases for CARP I/O functions (T1j).

Targets ``imatools.io.carp_io`` (to be created by migration T2c2).

Functions characterised:
  - ``read_pts``          — loadtxt float array from ``.pts`` file
  - ``read_elem``         — loadtxt int array from ``.elem`` file (Tr type, with/without tags)
  - ``read_lon``          — loadtxt float array from ``.lon`` file
  - ``saveToCarpTxt``     — write pts + elem, captured via read-back of written text
  - ``readParsePts``      — validated read of ``.pts``, returns (nodes, numNodes)

Intentionally SKIPPED (master bugs, no golden):
  - ``readParseElem``  — calls ``read_elem(default el_type='Tt')`` on a Tr file → column OOB
  - ``loadCarpMesh``   — same root cause; broken for triangle meshes

All inputs are derived from the shared ``_fixtures.py`` builders so the
capture env and test env see byte-identical arrays.
"""

from __future__ import annotations

import os
import tempfile

import _fixtures as fx
from _capture_golden import CaptureCase
from common import ioutils

# ---------------------------------------------------------------------------
# Shared fixture: write CARP mesh files once (per import)
# ---------------------------------------------------------------------------

_TMPDIR = tempfile.mkdtemp()
_BASE = str(fx.write_carp_mesh(_TMPDIR, "testmesh"))


# ---------------------------------------------------------------------------
# saveToCarpTxt helpers
# The function writes two files: <mshname>.pts and <mshname>.elem.
# Capture their text content (list of lines) so the golden is stable.
# ---------------------------------------------------------------------------


def _capture_save_to_carp_txt_pts():
    """Write pts+elem via saveToCarpTxt, read back the .pts file as a list of lines."""
    pts, elem, _region, _lon = fx.carp_mesh()
    out_dir = tempfile.mkdtemp()
    mshname = os.path.join(out_dir, "out")
    ioutils.saveToCarpTxt(pts, elem, mshname)
    with open(mshname + ".pts", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh.readlines()]


def _capture_save_to_carp_txt_elem():
    """Write pts+elem via saveToCarpTxt, read back the .elem file as a list of lines."""
    pts, elem, _region, _lon = fx.carp_mesh()
    out_dir = tempfile.mkdtemp()
    mshname = os.path.join(out_dir, "out")
    ioutils.saveToCarpTxt(pts, elem, mshname)
    with open(mshname + ".elem", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh.readlines()]


# ---------------------------------------------------------------------------
# readParsePts helper
# Returns (nodes_array, numNodes); capture nodes as npy, count as json.
# ---------------------------------------------------------------------------


def _read_parse_pts_nodes():
    nodes, _num_nodes = ioutils.readParsePts(_BASE + ".pts")
    return nodes


def _read_parse_pts_count():
    _nodes, num_nodes = ioutils.readParsePts(_BASE + ".pts")
    return num_nodes


CASES = [
    # ------------------------------------------------------------------
    # read_pts
    # ------------------------------------------------------------------
    CaptureCase(
        name="carp/read_pts",
        func=ioutils.read_pts,
        args=(_BASE + ".pts",),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # read_elem (Tr type, no tags)
    # ------------------------------------------------------------------
    CaptureCase(
        name="carp/read_elem_tr_notags",
        func=ioutils.read_elem,
        args=(_BASE + ".elem",),
        kwargs={"el_type": "Tr", "tags": False},
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # read_elem (Tr type, with tags)
    # ------------------------------------------------------------------
    CaptureCase(
        name="carp/read_elem_tr_tags",
        func=ioutils.read_elem,
        args=(_BASE + ".elem",),
        kwargs={"el_type": "Tr", "tags": True},
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # read_lon
    # ------------------------------------------------------------------
    CaptureCase(
        name="carp/read_lon",
        func=ioutils.read_lon,
        args=(_BASE + ".lon",),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # saveToCarpTxt — captured via read-back of written text
    # ------------------------------------------------------------------
    CaptureCase(
        name="carp/saveToCarpTxt_pts_lines",
        func=_capture_save_to_carp_txt_pts,
        fmt="json",
    ),
    CaptureCase(
        name="carp/saveToCarpTxt_elem_lines",
        func=_capture_save_to_carp_txt_elem,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # readParsePts — nodes array + count
    # ------------------------------------------------------------------
    CaptureCase(
        name="carp/readParsePts_nodes",
        func=_read_parse_pts_nodes,
        fmt="npy",
    ),
    CaptureCase(
        name="carp/readParsePts_count",
        func=_read_parse_pts_count,
        fmt="json",
    ),
]
