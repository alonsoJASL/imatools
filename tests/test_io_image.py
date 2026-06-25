"""Characterization tests for ``imatools.io.image_io`` (T1h / T2a4).

All tests import from the TARGET location ``imatools.io.image_io``.

T2a4 migration status
---------------------
All 13 ``xfail`` markers have been removed.  7 functions were added to
``io/image_io.py`` from master ``itktools`` (with the Cat-A SetSpacing fix in
``load_nrrd_image``).  The 3 colliding-name tests were re-pointed to dev's
behaviour (see notes per test).

Re-pointed tests (dev versions WIN, per Jose's decision)
---------------------------------------------------------
* ``test_get_nrrd_header``  — dev's ``get_nrrd_header`` accepts ``Union[str,
  Path]`` and raises ``FileNotFoundError`` for missing files; the returned dict
  structure is identical to master's so the existing golden is still valid.
  No body change needed beyond xfail removal.

* ``test_save_image_dir``  — dev's ``save_image(contract_or_image, output_path,
  overwrite=False)`` does NOT accept ``name=`` or ``manual_ow=`` kwargs.
  Rewritten to pass the full output path and ``overwrite=True`` instead.
  The image content / spacing are unchanged so the master golden still applies.

* ``test_save_image_path``  — same reason: ``manual_ow=True`` replaced by
  ``overwrite=True``.  The master golden still applies.

Functions characterised (master ``common/itktools.py`` → ``io/image_io``):
  - load_image_as_np
  - load_image
  - load_nrrd_base
  - get_nrrd_header
  - load_nrrd_image       (INTENT STUB — Cat-A fix applied; no golden)
  - save_image
  - fix_header_and_save
  - convert_to_inr
  - convert_from_inr
  - pointfile_to_image

Golden values were captured from master via::

    M=~/dev/python/imatools.worktrees/master
    ~/opt/anaconda3/bin/conda run -n imatools env \\
        PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module image_io --out tests/golden

Comparison notes
----------------
* All goldens are **json** (dict with keys ``array``, ``spacing``, ``origin``,
  ``direction``; or specialised dicts for meta/header cases).
* Array comparisons use ``np.testing.assert_array_equal`` (integer pixel data)
  or ``np.testing.assert_allclose`` (float fields like spacing, origin, direction).
* File I/O tests create their own ``tempfile.TemporaryDirectory()`` — independent
  of the capture-time temp paths.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import _fixtures as fx
import nrrd
import numpy as np
import pytest
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reduce_sitk_image(im: sitk.Image) -> Dict[str, Any]:
    """Reduce a sitk.Image to a dict matching the captured golden format."""
    return {
        "array": sitk.GetArrayFromImage(im).tolist(),
        "spacing": list(im.GetSpacing()),
        "origin": list(im.GetOrigin()),
        "direction": list(im.GetDirection()),
    }


def _reduce_nrrd_header(header) -> Dict[str, Any]:
    """Convert NRRD header to a dict matching the captured golden format."""
    out = {}
    for k, v in header.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
        else:
            out[k] = v
    return out


def _is_numeric_list(val) -> bool:
    """Return True if *val* is a list/tuple with no string elements."""
    if not isinstance(val, (list, tuple)):
        return False
    return all(not isinstance(x, str) for x in val)


def _assert_image_dict_equal(result: Dict[str, Any], golden: Dict[str, Any]) -> None:
    """Assert that a reduced image dict matches the golden.

    * ``array`` (list of int)  → array_equal
    * ``spacing``, ``origin``, ``direction`` (list of float) → allclose
    """
    np.testing.assert_array_equal(
        np.asarray(result["array"]),
        np.asarray(golden["array"]),
        err_msg="pixel array mismatch",
    )
    np.testing.assert_allclose(result["spacing"], golden["spacing"], rtol=1e-6, err_msg="spacing")
    np.testing.assert_allclose(result["origin"], golden["origin"], rtol=1e-6, err_msg="origin")
    np.testing.assert_allclose(
        result["direction"], golden["direction"], rtol=1e-6, err_msg="direction"
    )


def _assert_header_value(key, result_val, golden_val) -> None:
    """Assert a single NRRD header value.

    Numeric lists → ``assert_allclose``; string lists and scalars → ``==``.
    This handles the fact that some header keys (e.g. ``kinds``) contain lists
    of strings which ``assert_allclose`` cannot process.
    """
    if _is_numeric_list(golden_val):
        np.testing.assert_allclose(
            np.asarray(result_val),
            np.asarray(golden_val),
            rtol=1e-6,
            err_msg=f"header key {key!r}",
        )
    else:
        assert result_val == golden_val, f"header key {key!r}: {result_val!r} != {golden_val!r}"


# ---------------------------------------------------------------------------
# Fixtures written at test time
# ---------------------------------------------------------------------------


def _write_nii(tmpdir: Path) -> str:
    """Write the synthetic label image as a .nii.gz and return its path."""
    path = str(tmpdir / "label.nii.gz")
    sitk.WriteImage(fx.label_image(), path)
    return path


def _write_nrrd(tmpdir: Path) -> str:
    """Write the synthetic label image as a .nrrd and return its path."""
    path = str(tmpdir / "label.nrrd")
    sitk.WriteImage(fx.label_image(), path)
    return path


def _write_oblique_nrrd(tmpdir: Path) -> str:
    """Write a NRRD with non-axis-aligned space directions (for fix_header_and_save)."""
    path = str(tmpdir / "label_oblique.nrrd")
    data = sitk.GetArrayFromImage(fx.label_image())
    header = {
        "type": "uint8",
        "dimension": 3,
        "space": "left-posterior-superior",
        "sizes": list(data.shape),
        "space directions": np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
        "space origin": np.array([0.0, 0.0, 0.0]),
        "endian": "little",
        "encoding": "raw",
    }
    nrrd.write(path, data, header)
    return path


def _write_point_json(tmpdir: Path) -> str:
    """Write a JSON point file and return its path."""
    path = str(tmpdir / "points.json")
    points_dict = {
        "point_0": [3.0, 3.0, 3.0],
        "point_1": [7.0, 7.0, 7.0],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(points_dict, fh)
    return path


# ---------------------------------------------------------------------------
# load_image_as_np
# ---------------------------------------------------------------------------


def test_load_image_as_np_array(golden):
    from imatools.io.image_io import load_image_as_np

    with tempfile.TemporaryDirectory() as tmp:
        nii_path = _write_nii(Path(tmp))
        arr, _origin, _size = load_image_as_np(nii_path)
        expected = golden("image_io/load_image_as_np_array")
        np.testing.assert_array_equal(arr, np.asarray(expected))


def test_load_image_as_np_meta(golden):
    from imatools.io.image_io import load_image_as_np

    with tempfile.TemporaryDirectory() as tmp:
        nii_path = _write_nii(Path(tmp))
        _arr, origin, im_size = load_image_as_np(nii_path)
        expected = golden("image_io/load_image_as_np_meta")
        assert list(origin) == pytest.approx(expected["origin"])
        assert list(im_size) == list(expected["im_size"])


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


def test_load_image(golden):
    from imatools.io.image_io import load_image

    with tempfile.TemporaryDirectory() as tmp:
        nii_path = _write_nii(Path(tmp))
        # Call with master's positional signature: load_image(path_to_file, ext='nii')
        # Dev version may require return_contract=False to get a sitk.Image.
        result = load_image(nii_path)
        # Accept either a sitk.Image or an ImageContract with get_sitk_image()
        if not isinstance(result, sitk.Image):
            result = result.get_sitk_image()
        reduced = _reduce_sitk_image(result)
        expected = golden("image_io/load_image")
        _assert_image_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# load_nrrd_base
# ---------------------------------------------------------------------------


def test_load_nrrd_base_array(golden):
    from imatools.io.image_io import load_nrrd_base

    with tempfile.TemporaryDirectory() as tmp:
        nrrd_path = _write_nrrd(Path(tmp))
        data, _header = load_nrrd_base(nrrd_path)
        expected = golden("image_io/load_nrrd_base_array")
        np.testing.assert_array_equal(data, np.asarray(expected))


def test_load_nrrd_base_header(golden):
    from imatools.io.image_io import load_nrrd_base

    with tempfile.TemporaryDirectory() as tmp:
        nrrd_path = _write_nrrd(Path(tmp))
        _data, header = load_nrrd_base(nrrd_path)
        reduced = _reduce_nrrd_header(header)
        expected = golden("image_io/load_nrrd_base_header")
        assert set(reduced.keys()) == set(
            expected.keys()
        ), f"header key mismatch: got {set(reduced.keys())}, expected {set(expected.keys())}"
        for key in expected:
            _assert_header_value(key, reduced[key], expected[key])


# ---------------------------------------------------------------------------
# get_nrrd_header
#
# Re-pointed to dev's behaviour (T2a4):
#   Dev's get_nrrd_header(path) accepts Union[str, Path] and raises
#   FileNotFoundError for missing files.  The returned dict structure is
#   identical to master's (both call nrrd.read()), so the master golden is
#   still valid and no golden file was changed.
# ---------------------------------------------------------------------------


def test_get_nrrd_header(golden):
    from imatools.io.image_io import get_nrrd_header

    with tempfile.TemporaryDirectory() as tmp:
        nrrd_path = _write_nrrd(Path(tmp))
        header = get_nrrd_header(nrrd_path)
        reduced = _reduce_nrrd_header(header)
        expected = golden("image_io/get_nrrd_header")
        assert set(reduced.keys()) == set(
            expected.keys()
        ), f"header key mismatch: got {set(reduced.keys())}, expected {set(expected.keys())}"
        for key in expected:
            _assert_header_value(key, reduced[key], expected[key])


# ---------------------------------------------------------------------------
# load_nrrd_image — INTENT STUB
#
# Master's implementation has a bug: it calls
#   sitk_image.SetSpacing(header.get('space directions', (1, 1, 1)))
# where 'space directions' is a (3,3) ndarray written by SimpleITK.  SITK
# rejects anything that isn't a sequence of scalars, so the call raises
# TypeError.  No golden is committed for this function; the test just
# asserts the Cat-A fix works (dev should NOT crash).
# ---------------------------------------------------------------------------


def test_load_nrrd_image_intent_stub():
    """Intent stub: dev's load_nrrd_image should succeed where master crashes.

    Master raises ``TypeError`` because it passes a (3,3) 'space directions'
    matrix to ``sitk.Image.SetSpacing``.  The dev implementation must fix this.
    """
    from imatools.io.image_io import load_nrrd_image

    with tempfile.TemporaryDirectory() as tmp:
        nrrd_path = _write_nrrd(Path(tmp))
        # Should NOT raise; if it does, the migration has not fixed the bug.
        result = load_nrrd_image(nrrd_path)
        assert isinstance(result, sitk.Image), "expected sitk.Image"
        assert result.GetSize() == (12, 12, 12), "size mismatch"


# ---------------------------------------------------------------------------
# save_image
#
# Re-pointed to dev's behaviour (T2a4):
#   Master's save_image(image, dir_or_path, name=None, manual_ow=False)
#   Dev's  save_image(contract, output_path, overwrite=False)
#
#   test_save_image_dir: pass full path (dir / name) instead of dir + name=
#   test_save_image_path: replace manual_ow=True with overwrite=True
#
#   The golden files are unchanged because the saved image content and
#   metadata (array, spacing, origin, direction) are identical for both APIs.
# ---------------------------------------------------------------------------


def test_save_image_dir(golden):
    """save_image(image, full_path, overwrite=True) round-trip.

    Re-pointed from master's ``save_image(image, dir, name=..., manual_ow=True)``
    to dev's ``save_image(image, output_path, overwrite=True)``.  The golden
    file is unchanged — image content is the same.
    """
    from imatools.io.image_io import save_image

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_name = "saved_label.nii.gz"
        # Dev API: pass the full output path, overwrite=True (no name= kwarg)
        save_image(fx.label_image(), str(tmp_path / out_name), overwrite=True)
        back = sitk.ReadImage(str(tmp_path / out_name))
        reduced = _reduce_sitk_image(back)
        expected = golden("image_io/save_image_dir")
        _assert_image_dict_equal(reduced, expected)


def test_save_image_path(golden):
    """save_image(image, full_path, overwrite=True) round-trip (name=None).

    Re-pointed from master's ``save_image(image, path, manual_ow=True)``
    to dev's ``save_image(image, path, overwrite=True)``.  The golden
    file is unchanged — image content is the same.
    """
    from imatools.io.image_io import save_image

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "saved_label_path.nii.gz")
        # Dev API: overwrite=True instead of manual_ow=True
        save_image(fx.label_image(), out_path, overwrite=True)
        back = sitk.ReadImage(out_path)
        reduced = _reduce_sitk_image(back)
        expected = golden("image_io/save_image_path")
        _assert_image_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# fix_header_and_save
# ---------------------------------------------------------------------------


def test_fix_header_and_save(golden):
    """fix_header_and_save round-trip: oblique header → axis-aligned."""
    from imatools.io.image_io import fix_header_and_save

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        in_path = _write_oblique_nrrd(tmp_path)
        out_path = str(tmp_path / "label_fixed.nrrd")
        fix_header_and_save(in_path, out_path)
        _, hdr = nrrd.read(out_path)
        reduced = _reduce_nrrd_header(hdr)
        expected = golden("image_io/fix_header_and_save")
        assert set(reduced.keys()) == set(
            expected.keys()
        ), f"header key mismatch: got {set(reduced.keys())}, expected {set(expected.keys())}"
        for key in expected:
            _assert_header_value(key, reduced[key], expected[key])


# ---------------------------------------------------------------------------
# convert_to_inr
# ---------------------------------------------------------------------------


def test_convert_to_inr(golden):
    """convert_to_inr: write .inr, read back via convert_from_inr, compare array."""
    from imatools.io.image_io import convert_from_inr, convert_to_inr

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "label.inr")
        convert_to_inr(fx.label_image(), out_path)
        back = convert_from_inr(out_path)
        reduced = _reduce_sitk_image(back)
        expected = golden("image_io/convert_to_inr")
        _assert_image_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# convert_from_inr
# ---------------------------------------------------------------------------


def test_convert_from_inr(golden):
    """convert_from_inr round-trip: write .inr then read back."""
    from imatools.io.image_io import convert_from_inr, convert_to_inr

    with tempfile.TemporaryDirectory() as tmp:
        inr_path = str(Path(tmp) / "label_roundtrip.inr")
        convert_to_inr(fx.label_image(), inr_path)
        result = convert_from_inr(inr_path)
        reduced = _reduce_sitk_image(result)
        expected = golden("image_io/convert_from_inr")
        _assert_image_dict_equal(reduced, expected)


# ---------------------------------------------------------------------------
# pointfile_to_image
# ---------------------------------------------------------------------------


def test_pointfile_to_image(golden):
    """pointfile_to_image: write .nii + JSON points, call, compare reduced image."""
    from imatools.io.image_io import pointfile_to_image

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        nii_path = _write_nii(tmp_path)
        json_path = _write_point_json(tmp_path)
        result = pointfile_to_image(nii_path, json_path, label=5, girth=1)
        reduced = _reduce_sitk_image(result)
        expected = golden("image_io/pointfile_to_image")
        _assert_image_dict_equal(reduced, expected)
