"""Capture cases for path-helper functions (T1j).

Targets ``imatools.io.paths`` (to be created by migration T2c3).

Functions characterised:
  - ``fullfile``           — pure join (no I/O)
  - ``ext``                — filename + extension helper (no I/O)
  - ``num2padstr``         — zero-padded string (no I/O)
  - ``mkdirplus``          — creates directories, returns path
  - ``searchFileByType``   — glob for files by prefix + extension
  - ``get_subfolders``     — list subdirectories
  - ``find_file``          — locate file by name fragment + optional extension

For filesystem-dependent cases a fixed-structure temporary directory is seeded
once at import time.  Goldens record STABLE reduced values (sorted basenames,
boolean existence, constant strings) so they are reproducible across machines.
The matching tests reproduce the same reduction from a freshly-seeded tempdir.

NOT in scope: ``check_file_exists`` (relocate decision belongs to T2c3).
"""

from __future__ import annotations

import os
import tempfile

from _capture_golden import CaptureCase
from common import ioutils

# ---------------------------------------------------------------------------
# Shared temporary filesystem tree (seeded once at import time)
#
#   <TMPDIR>/
#     alpha/          <- subfolder
#     beta/           <- subfolder
#     mesh1.vtk
#     mesh2.vtk
#     image.nii
#     mesh1.nii       <- same stem as mesh1.vtk (for disambiguation test)
# ---------------------------------------------------------------------------

_TMPDIR = tempfile.mkdtemp()
os.makedirs(os.path.join(_TMPDIR, "alpha"))
os.makedirs(os.path.join(_TMPDIR, "beta"))
for _fname in ("mesh1.vtk", "mesh2.vtk", "image.nii", "mesh1.nii"):
    open(os.path.join(_TMPDIR, _fname), "w").close()


# ---------------------------------------------------------------------------
# Reduction helpers (stable, not path-dependent)
# ---------------------------------------------------------------------------


def _mkdirplus_creates():
    """mkdirplus creates a nested dir; capture boolean existence as the golden."""
    result = ioutils.mkdirplus(_TMPDIR, "newdir", "leaf")
    return os.path.isdir(result)


def _search_file_by_type_vtk():
    """Sorted basenames of all *.vtk files."""
    files = ioutils.searchFileByType(_TMPDIR, extension="vtk")
    return sorted(os.path.basename(f) for f in files)


def _search_file_by_type_prefix():
    """Sorted basenames of mesh*.vtk files."""
    files = ioutils.searchFileByType(_TMPDIR, prefix="mesh", extension="vtk")
    return sorted(os.path.basename(f) for f in files)


def _get_subfolders_sorted():
    """Sorted basenames of subdirectories."""
    subs = ioutils.get_subfolders(_TMPDIR)
    return sorted(os.path.basename(s) for s in subs)


def _find_file_single():
    """Find unique file by name fragment; return its basename."""
    found = ioutils.find_file(_TMPDIR, "image")
    return os.path.basename(found) if found else ""


def _find_file_with_ext():
    """Disambiguate mesh1.vtk vs mesh1.nii using extension hint; return basename."""
    found = ioutils.find_file(_TMPDIR, "mesh1", extension="nii")
    return os.path.basename(found) if found else ""


def _find_file_missing():
    """Return value when file does not exist."""
    return ioutils.find_file(_TMPDIR, "nonexistent_xyz")


CASES = [
    # ------------------------------------------------------------------
    # fullfile — pure join
    # ------------------------------------------------------------------
    CaptureCase(
        name="paths/fullfile_3parts",
        func=ioutils.fullfile,
        args=("a", "b", "c"),
        fmt="json",
    ),
    CaptureCase(
        name="paths/fullfile_2parts",
        func=ioutils.fullfile,
        args=("dir", "file.txt"),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # ext — filename extension helper
    # ------------------------------------------------------------------
    CaptureCase(
        name="paths/ext_add",
        func=ioutils.ext,
        args=("file", "txt"),
        fmt="json",
    ),
    CaptureCase(
        name="paths/ext_already_has",
        func=ioutils.ext,
        args=("file.txt", "txt"),
        fmt="json",
    ),
    CaptureCase(
        name="paths/ext_dot_prefix",
        func=ioutils.ext,
        args=("file", ".nii"),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # num2padstr — zero-padded string
    # ------------------------------------------------------------------
    CaptureCase(
        name="paths/num2padstr_default",
        func=ioutils.num2padstr,
        args=(5,),
        fmt="json",
    ),
    CaptureCase(
        name="paths/num2padstr_wide",
        func=ioutils.num2padstr,
        args=(42,),
        kwargs={"padding": 5},
        fmt="json",
    ),
    CaptureCase(
        name="paths/num2padstr_overflow",
        func=ioutils.num2padstr,
        args=(1000,),
        kwargs={"padding": 3},
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # mkdirplus — creates directories
    # ------------------------------------------------------------------
    CaptureCase(
        name="paths/mkdirplus_creates",
        func=_mkdirplus_creates,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # searchFileByType — glob by extension / prefix
    # ------------------------------------------------------------------
    CaptureCase(
        name="paths/searchFileByType_vtk",
        func=_search_file_by_type_vtk,
        fmt="json",
    ),
    CaptureCase(
        name="paths/searchFileByType_prefix",
        func=_search_file_by_type_prefix,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # get_subfolders — list subdirectories
    # ------------------------------------------------------------------
    CaptureCase(
        name="paths/get_subfolders_sorted",
        func=_get_subfolders_sorted,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # find_file — locate file by name fragment
    # ------------------------------------------------------------------
    CaptureCase(
        name="paths/find_file_single",
        func=_find_file_single,
        fmt="json",
    ),
    CaptureCase(
        name="paths/find_file_with_ext",
        func=_find_file_with_ext,
        fmt="json",
    ),
    CaptureCase(
        name="paths/find_file_missing",
        func=_find_file_missing,
        fmt="json",
    ),
]
