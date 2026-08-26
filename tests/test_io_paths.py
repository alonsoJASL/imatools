"""Characterization tests for ``imatools.io.paths`` (T1j).

All tests import from the TARGET location ``imatools.io.paths`` (created by
migration task T2c3).

Golden values were captured from master via::

    ~/opt/anaconda3/bin/conda run -n imatools env \\
        PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module paths --out tests/golden

where ``M = ~/dev/python/imatools.worktrees/master``.

For filesystem-dependent functions (``get_subfolders``, ``find_file``) the
tests build a fixed-structure temporary directory at test time and reduce the
results to stable values (sorted basenames, constant strings) that match the
captured goldens.

NOT characterised here: ``check_file_exists`` (relocation is a T2c3 concern).
The legacy ``fullfile`` / ``mkdirplus`` / ``searchFileByType`` helpers were
removed (no live consumer); their cases were retired with them.
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
# get_subfolders — list subdirectories
# ---------------------------------------------------------------------------


def test_get_subfolders_sorted(golden):
    from imatools.io.paths import get_subfolders

    tmpdir = tempfile.mkdtemp()
    _seed_tmpdir(tmpdir)
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
