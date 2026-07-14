"""CARP mesh I/O functions.

Migrated from ``imatools.common.ioutils`` (T2c2); that shim module was deleted
in M2 — this is now the sole home.

M3-C1 fix: ``readParseElem`` / ``loadCarpMesh`` now detect the element type from
the ``.elem`` file (master hardcoded ``el_type='Tt'``, which broke triangle meshes).

Cat-B bug still preserved (per Wave-2 bug policy; fix deferred to future_work):
  - ``saveToCarpTxt`` hardcodes ``fmt='Tr %d %d %d 1'`` (element tag always 1).
"""

from __future__ import annotations

import sys

import numpy as np

# ---------------------------------------------------------------------------
# ``fullfile`` (a path helper) lives in ``io.paths``; accessed via a lazy
# call-time import to avoid circular-import issues (M2c — was routed through the
# ``common.ioutils`` shim, now gone). ``getTotal`` used to be reached the same
# way but now lives here directly (M2b — a CARP-file header helper).
# ---------------------------------------------------------------------------


def _paths():
    import imatools.io.paths as _p  # noqa: PLC0415

    return _p


# ---------------------------------------------------------------------------
# Module-level constant (moved from ioutils together with read_elem)
# ---------------------------------------------------------------------------

ELEM_TYPES = ["Tt", "Tr", "Ln"]


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------


def read_pts(filename):
    print(f"Reading: {filename}")
    return np.loadtxt(filename, dtype=float, skiprows=1)


def read_elem(filename, el_type="Tt", tags=True):
    if el_type not in ELEM_TYPES:
        raise Exception("element type not recognised. Accepted: Tt, Tr, Ln")

    cols_notags_dic = {"Tt": (1, 2, 3, 4), "Tr": (1, 2, 3), "Ln": (1, 2)}
    cols = cols_notags_dic[el_type]
    if tags:
        # add tags column (largest + 1)
        cols += (cols[-1] + 1,)

    return np.loadtxt(filename, dtype=int, skiprows=1, usecols=cols)


def read_lon(filename):
    print(f"Reading: {filename}")
    return np.loadtxt(filename, dtype=float, skiprows=1)


def _detect_elem_type(elFname):  # noqa: N803
    """Return the element type (``Tr``/``Tt``/``Ln``) from a CARP ``.elem`` file.

    Reads the first token of the first data row (after the count header). Falls
    back to ``Tt`` if the row has no recognised type token.
    """
    with open(elFname, encoding="utf-8") as f:
        f.readline()  # skip the count header
        first = f.readline().split()
    token = first[0] if first else "Tt"
    return token if token in ELEM_TYPES else "Tt"


def getTotal(fname):  # noqa: N802
    """Get the total count declared on the first line of a CARP .pts/.elem file.

    Migrated verbatim from ``common.ioutils`` (M2b): its only callers are this
    module's ``readParsePts``/``readParseElem``. Complementary to ``read_pts``/
    ``read_elem`` (which read the data rows via ``skiprows=1``) — this reads the
    declared header count so the parsers can validate it against the data.
    """
    try:
        with open(fname, encoding="utf-8") as f:
            numNodes = int(f.readline().strip())  # noqa: N806
    except Exception:
        print("[getTotal] Error - file not found")
        sys.exit(-1)

    return numNodes


# ---------------------------------------------------------------------------
# Higher-level parsers
# ---------------------------------------------------------------------------


def readParsePts(ptsFname):  # noqa: N802,N803
    """Read parse CARP point files."""
    numNodes = getTotal(ptsFname)  # noqa: N806
    nodes = read_pts(ptsFname)

    if numNodes != len(nodes):
        print("Error in file")
        raise Exception("Error in file")

    return nodes, numNodes


def readParseElem(elFname):  # noqa: N802,N803
    """Read and parse CARP element file.

    Detects the element type (``Tr``/``Tt``/``Ln``) from the file's first data
    row so triangle ``.elem`` files parse correctly (M3-C1 fix — master hardcoded
    ``el_type='Tt'``, which requested a non-existent tet column on triangle files
    and raised ``ValueError``).
    """
    nElem = getTotal(elFname)  # noqa: N806
    el = read_elem(elFname, el_type=_detect_elem_type(elFname))

    if nElem != len(el):
        print("Error in file")
        raise Exception("Error in file")

    return el, nElem


def loadCarpMesh(mshname, directory=None):  # noqa: N802
    """Load a CARP mesh — supports triangle (``Tr``) and tetrahedral (``Tt``) elements.

    Returns ``(pts, elem, region)``: ``pts`` an ``(N, 3)`` float array, ``elem`` a
    list of per-element connectivity lists, ``region`` an int array of element tags.
    (M3-C1 fix — the element type is detected from the file, and the parse works on
    ``read_elem``'s integer connectivity+tag output.)
    """
    paths = _paths()

    if directory is not None:
        ptsname = paths.fullfile(directory, mshname + ".pts")
        elemname = paths.fullfile(directory, mshname + ".elem")
    else:
        ptsname = mshname + ".pts"
        elemname = mshname + ".elem"

    pts, nPts = readParsePts(ptsname)  # noqa: F841,N806
    el, nElem = readParseElem(elemname)  # noqa: F841,N806

    # read_elem returns rows of ``[n0, n1, ..., tag]`` (int); the last column is
    # the element/region tag, the rest is connectivity.
    el = np.atleast_2d(el)
    elem = [row[:-1].tolist() for row in el]
    region = np.asarray([row[-1] for row in el], dtype=int)

    return pts, elem, region


def saveToCarpTxt(pts, el, mshname):  # noqa: N802
    """Save CARP mesh to text files.

    Cat-B bug preserved: element format hardcodes tag=1 (``fmt='Tr %d %d %d 1'``).
    """
    np.savetxt(mshname + ".pts", pts, header=str(len(pts)), comments="", fmt="%6.12f")
    np.savetxt(mshname + ".elem", el, header=str(len(el)), comments="", fmt="Tr %d %d %d 1")
