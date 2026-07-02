"""CARP mesh I/O functions.

Migrated from ``imatools.common.ioutils`` (T2c2).  Legacy callers that import
from ``imatools.common.ioutils`` continue to work via the re-export shim at
the bottom of that module.

Cat-B bugs preserved verbatim (per Wave-2 bug policy):
  - ``readParseElem`` / ``loadCarpMesh`` call ``read_elem`` with the default
    ``el_type='Tt'`` which requests column index 5 on triangle ``.elem`` files
    (only 5 columns, 0-4) → raises ``ValueError`` for triangle meshes.
  - ``saveToCarpTxt`` hardcodes ``fmt='Tr %d %d %d 1'`` (element tag always 1).
"""

from __future__ import annotations

import sys

import numpy as np

# ---------------------------------------------------------------------------
# ``fullfile`` (a path helper shimmed in ioutils → io.paths) is accessed via a
# lazy call-time import to avoid circular-import issues (carp_io may be imported
# before ioutils is fully initialised).  ``getTotal`` used to be reached the same
# way but now lives here directly (M2b — it is a CARP-file header helper with no
# other caller).
# ---------------------------------------------------------------------------


def _ioutils():
    import imatools.common.ioutils as io  # noqa: PLC0415

    return io


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

    Cat-B bug preserved: calls ``read_elem`` with the default ``el_type='Tt'``
    which raises ``ValueError`` on triangle ``.elem`` files.
    """
    nElem = getTotal(elFname)  # noqa: N806
    el = read_elem(elFname)

    if nElem != len(el):
        print("Error in file")
        raise Exception("Error in file")

    return el, nElem


def loadCarpMesh(mshname, directory=None):  # noqa: N802
    """Load CARP mesh. Supports for triangle (Tr) and tetrahedral (Tt) meshes.

    Cat-B bug preserved: calls ``readParseElem`` which in turn calls
    ``read_elem`` with the default ``el_type='Tt'``, raising ``ValueError``
    on triangle meshes.
    """
    io = _ioutils()

    if directory is not None:
        ptsname = io.fullfile(directory, mshname + ".pts")
        elemname = io.fullfile(directory, mshname + ".elem")
    else:
        ptsname = mshname + ".pts"
        elemname = mshname + ".elem"

    pts, nPts = readParsePts(ptsname)  # noqa: F841,N806
    el, nElem = readParseElem(elemname)  # noqa: F841,N806

    elem = list()
    for e in el:
        nel = 4 if e[0] == "Tr" else 5
        elem_before = e[1:nel]
        elem.append([int(ex.strip()) for ex in elem_before])

    region_before = [e[-1] for e in el]
    region = [int(x.strip()) for x in region_before]

    return pts, elem, np.asarray(region, dtype=int)


def saveToCarpTxt(pts, el, mshname):  # noqa: N802
    """Save CARP mesh to text files.

    Cat-B bug preserved: element format hardcodes tag=1 (``fmt='Tr %d %d %d 1'``).
    """
    np.savetxt(mshname + ".pts", pts, header=str(len(pts)), comments="", fmt="%6.12f")
    np.savetxt(mshname + ".elem", el, header=str(len(el)), comments="", fmt="Tr %d %d %d 1")
