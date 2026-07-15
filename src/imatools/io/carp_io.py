"""CARP mesh I/O functions.

Migrated from ``imatools.common.ioutils`` (T2c2); that shim module was deleted
in M2 — this is now the sole home.

M3-C1 fix: ``readParseElem`` / ``loadCarpMesh`` now detect the element type from
the ``.elem`` file (master hardcoded ``el_type='Tt'``, which broke triangle meshes).

Cat-B bug still preserved (per Wave-2 bug policy; fix deferred to future_work):
  - ``saveToCarpTxt`` hardcodes ``fmt='Tr %d %d %d 1'`` (element tag always 1).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constant (moved from ioutils together with read_elem)
# ---------------------------------------------------------------------------

ELEM_TYPES = ["Tt", "Tr", "Ln"]


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------


def read_pts(filename: Union[str, Path]) -> np.ndarray:
    logger.info(f"Reading: {filename}")
    return np.loadtxt(filename, dtype=float, skiprows=1)


def read_elem(filename: Union[str, Path], el_type="Tt", tags=True) -> np.ndarray:
    if el_type not in ELEM_TYPES:
        raise Exception("element type not recognised. Accepted: Tt, Tr, Ln")

    cols_notags_dic = {"Tt": (1, 2, 3, 4), "Tr": (1, 2, 3), "Ln": (1, 2)}
    cols = cols_notags_dic[el_type]
    if tags:
        # add tags column (largest + 1)
        cols += (cols[-1] + 1,)

    return np.loadtxt(filename, dtype=int, skiprows=1, usecols=cols)


def read_lon(filename: Union[str, Path]) -> np.ndarray:
    logger.info(f"Reading: {filename}")
    return np.loadtxt(filename, dtype=float, skiprows=1)


def _detect_elem_type(elFname: Union[str, Path]):  # noqa: N803
    """Return the element type (``Tr``/``Tt``/``Ln``) from a CARP ``.elem`` file.

    Reads the first token of the first data row (after the count header). Falls
    back to ``Tt`` if the row has no recognised type token.
    """
    with open(elFname, encoding="utf-8") as f:
        f.readline()  # skip the count header
        first = f.readline().split()
    token = first[0] if first else "Tt"
    return token if token in ELEM_TYPES else "Tt"


def get_total(fname: Union[str, Path]) -> int:
    """Get the total count declared on the first line of a CARP .pts/.elem file.

    Migrated from ``common.ioutils`` (M2b): its only callers are this module's
    ``readParsePts``/``readParseElem``. Complementary to ``read_pts``/``read_elem``
    (which read the data rows via ``skiprows=1``) — this reads the declared header
    count so the parsers can validate it against the data.

    A missing file raises ``FileNotFoundError``. The migrated version logged and
    called ``sys.exit(-1)``, which killed the caller's process and left importers
    (CLIs, notebooks) no chance to handle it; raising is the deliberate change.
    """
    with open(fname, encoding="utf-8") as f:
        return int(f.readline().strip())


def getTotal(fname: Union[str, Path]) -> int:  # noqa: N802
    logger.warning("getTotal() is deprecated; use get_total() instead")
    return get_total(fname)


# ---------------------------------------------------------------------------
# Higher-level parsers
# ---------------------------------------------------------------------------


def readParsePts(ptsFname: Union[str, Path]):  # noqa: N802,N803
    """Read parse CARP point files."""
    numNodes = get_total(ptsFname)  # noqa: N806
    nodes = read_pts(ptsFname)

    if numNodes != len(nodes):
        logger.error("Error in file")
        raise Exception("Error in file")

    return nodes, numNodes


def readParseElem(elFname: Union[str, Path]):  # noqa: N802,N803
    """Read and parse CARP element file.

    Detects the element type (``Tr``/``Tt``/``Ln``) from the file's first data
    row so triangle ``.elem`` files parse correctly (M3-C1 fix — master hardcoded
    ``el_type='Tt'``, which requested a non-existent tet column on triangle files
    and raised ``ValueError``).
    """
    nElem = get_total(elFname)  # noqa: N806
    el = read_elem(elFname, el_type=_detect_elem_type(elFname))

    if nElem != len(el):
        logger.error("Error in file")
        raise Exception("Error in file")

    return el, nElem


def loadCarpMesh(mshname: Union[str, Path], directory: Union[str, Path, None] = None):  # noqa: N802
    """Load a CARP mesh — supports triangle (``Tr``) and tetrahedral (``Tt``) elements.

    Returns ``(pts, elem, region)``: ``pts`` an ``(N, 3)`` float array, ``elem`` a
    list of per-element connectivity lists, ``region`` an int array of element tags.
    (M3-C1 fix — the element type is detected from the file, and the parse works on
    ``read_elem``'s integer connectivity+tag output.)
    """
    base = Path(directory) / mshname if directory is not None else Path(mshname)
    ptsname = f"{base}.pts"
    elemname = f"{base}.elem"

    pts, nPts = readParsePts(ptsname)  # noqa: F841,N806
    el, nElem = readParseElem(elemname)  # noqa: F841,N806

    # read_elem returns rows of ``[n0, n1, ..., tag]`` (int); the last column is
    # the element/region tag, the rest is connectivity.
    el = np.atleast_2d(el)
    elem = [row[:-1].tolist() for row in el]
    region = np.asarray([row[-1] for row in el], dtype=int)

    return pts, elem, region


def saveToCarpTxt(pts, el, mshname: Union[str, Path]):  # noqa: N802
    """Save CARP mesh to text files.

    ``mshname`` is a basename with no extension; ``.pts``/``.elem`` are appended.

    Cat-B bug preserved: element format hardcodes tag=1 (``fmt='Tr %d %d %d 1'``).
    """
    base = Path(mshname)
    np.savetxt(f"{base}.pts", pts, header=str(len(pts)), comments="", fmt="%6.12f")
    np.savetxt(f"{base}.elem", el, header=str(len(el)), comments="", fmt="Tr %d %d %d 1")
