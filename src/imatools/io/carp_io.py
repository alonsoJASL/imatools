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

import numpy as np

# ---------------------------------------------------------------------------
# Helpers used by the CARP functions that stay in ioutils (getTotal, fullfile)
# are accessed via a lazy call-time import to avoid circular-import issues
# (carp_io may be imported before ioutils is fully initialised).
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


# ---------------------------------------------------------------------------
# Higher-level parsers
# ---------------------------------------------------------------------------


def readParsePts(ptsFname):  # noqa: N802,N803
    """Read parse CARP point files."""
    numNodes = _ioutils().getTotal(ptsFname)  # noqa: N806
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
    nElem = _ioutils().getTotal(elFname)  # noqa: N806
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
