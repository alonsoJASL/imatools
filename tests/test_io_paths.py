"""Characterization tests for ``imatools.io.paths`` (T1j).

All tests import from the TARGET location ``imatools.io.paths``.  That module
does not exist yet — it will be created by migration task T2c3.  Until then
every test is marked ``xfail(strict=False)`` so it is collected but does not
block CI.

Golden values were captured from master via::

    ~/opt/anaconda3/bin/conda run -n imatools env \\
        PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module paths --out tests/golden

where ``M = ~/dev/python/imatools.worktrees/master``.

For filesystem-dependent functions (``mkdirplus``, ``searchFileByType``,
``get_subfolders``, ``find_file``) the tests build a fixed-structure temporary
directory at test time and reduce the results to stable values (sorted basenames,
boolean existence, constant strings) that match the captured goldens.

NOT characterised here: ``check_file_exists`` (relocation is a T2c3 concern).
"""

from __future__ import annotations

import os
import tempfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_tmpdir(tmpdir: str) -> None:
    """Populate a temporary directory with a fixed, known structure.

    Structure::

        <tmpdir>/
          alpha/
          beta/
          mesh1.vtk
          mesh2.vtk
          image.nii
          mesh1.nii
    """
    os.makedirs(os.path.join(tmpdir, "alpha"))
    os.makedirs(os.path.join(tmpdir, "beta"))
    for fname in ("mesh1.vtk", "mesh2.vtk", "image.nii", "mesh1.nii"):
        open(os.path.join(tmpdir, fname), "w").close()


# ---------------------------------------------------------------------------
# fullfile — pure join
# ---------------------------------------------------------------------------


def test_fullfile_3parts(golden):
    from imatools.io.paths import fullfile

    result = fullfile("a", "b", "c")
    expected = golden("paths/fullfile_3parts")
    assert result == expected


def test_fullfile_2parts(golden):
    from imatools.io.paths import fullfile

    result = fullfile("dir", "file.txt")
    expected = golden("paths/fullfile_2parts")
    assert result == expected


# ---------------------------------------------------------------------------
# ext — filename extension helper
# ---------------------------------------------------------------------------


def test_ext_add(golden):
    from imatools.io.paths import ext

    result = ext("file", "txt")
    expected = golden("paths/ext_add")
    assert result == expected


def test_ext_already_has(golden):
    from imatools.io.paths import ext

    result = ext("file.txt", "txt")
    expected = golden("paths/ext_already_has")
    assert result == expected


def test_ext_dot_prefix(golden):
    from imatools.io.paths import ext

    result = ext("file", ".nii")
    expected = golden("paths/ext_dot_prefix")
    assert result == expected


# ---------------------------------------------------------------------------
# num2padstr — zero-padded string
# ---------------------------------------------------------------------------


def test_num2padstr_default(golden):
    from imatools.io.paths import num2padstr

    result = num2padstr(5)
    expected = golden("paths/num2padstr_default")
    assert result == expected


def test_num2padstr_wide(golden):
    from imatools.io.paths import num2padstr

    result = num2padstr(42, padding=5)
    expected = golden("paths/num2padstr_wide")
    assert result == expected


def test_num2padstr_overflow(golden):
    from imatools.io.paths import num2padstr

    result = num2padstr(1000, padding=3)
    expected = golden("paths/num2padstr_overflow")
    assert result == expected


# ---------------------------------------------------------------------------
# mkdirplus — creates directories, returns the path
# ---------------------------------------------------------------------------


def test_mkdirplus_creates(golden):
    from imatools.io.paths import mkdirplus

    tmpdir = tempfile.mkdtemp()
    result = mkdirplus(tmpdir, "newdir", "leaf")
    expected = golden("paths/mkdirplus_creates")
    assert os.path.isdir(result) == expected


# ---------------------------------------------------------------------------
# searchFileByType — glob by extension / prefix
# ---------------------------------------------------------------------------


def test_search_file_by_type_vtk(golden):
    from imatools.io.paths import searchFileByType

    tmpdir = tempfile.mkdtemp()
    _seed_tmpdir(tmpdir)
    files = searchFileByType(tmpdir, extension="vtk")
    result = sorted(os.path.basename(f) for f in files)
    expected = golden("paths/searchFileByType_vtk")
    assert result == expected


def test_search_file_by_type_prefix(golden):
    from imatools.io.paths import searchFileByType

    tmpdir = tempfile.mkdtemp()
    _seed_tmpdir(tmpdir)
    files = searchFileByType(tmpdir, prefix="mesh", extension="vtk")
    result = sorted(os.path.basename(f) for f in files)
    expected = golden("paths/searchFileByType_prefix")
    assert result == expected


# ---------------------------------------------------------------------------
# get_subfolders — list subdirectories
# ---------------------------------------------------------------------------


def test_get_subfolders_sorted(golden):
    from imatools.io.paths import get_subfolders, mkdirplus

    tmpdir = tempfile.mkdtemp()
    _seed_tmpdir(tmpdir)
    # The golden was captured from a shared tmpdir where mkdirplus already
    # created 'newdir/leaf' before get_subfolders ran.  Reproduce that here.
    mkdirplus(tmpdir, "newdir", "leaf")
    subs = get_subfolders(tmpdir)
    result = sorted(os.path.basename(s) for s in subs)
    expected = golden("paths/get_subfolders_sorted")
    assert result == expected


# ---------------------------------------------------------------------------
# find_file — locate file by name fragment
# ---------------------------------------------------------------------------


def test_find_file_single(golden):
    from imatools.io.paths import find_file

    tmpdir = tempfile.mkdtemp()
    _seed_tmpdir(tmpdir)
    found = find_file(tmpdir, "image")
    result = os.path.basename(found) if found else ""
    expected = golden("paths/find_file_single")
    assert result == expected


def test_find_file_with_ext(golden):
    from imatools.io.paths import find_file

    tmpdir = tempfile.mkdtemp()
    _seed_tmpdir(tmpdir)
    found = find_file(tmpdir, "mesh1", extension="nii")
    result = os.path.basename(found) if found else ""
    expected = golden("paths/find_file_with_ext")
    assert result == expected


def test_find_file_missing(golden):
    from imatools.io.paths import find_file

    tmpdir = tempfile.mkdtemp()
    result = find_file(tmpdir, "nonexistent_xyz")
    expected = golden("paths/find_file_missing")
    assert result == expected
