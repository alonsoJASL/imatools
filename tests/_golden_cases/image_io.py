"""Capture cases: itktools file I/O surface → ``io/image_io``.

Functions characterised (master ``common/itktools.py``):
  - load_image_as_np
  - load_image
  - load_nrrd_base
  - get_nrrd_header
  - load_nrrd_image
  - save_image          (writer — temp-write-then-read)
  - fix_header_and_save (writer — temp-write-then-read)
  - convert_to_inr      (writer — temp-write-then-read)
  - convert_from_inr    (reader — writes .inr via convert_to_inr, then reads back)
  - pointfile_to_image  (reader — writes .nii + .json point file, then reads back)

File-I/O pattern
~~~~~~~~~~~~~~~~
Readers: write a valid input file to a temp directory, then characterise the result.
Writers: call the function with a temp output path, read the written file back, treat
the reduced content as the golden.

All temp directories are created at module load time and kept alive for the module
lifetime (gc'd automatically when the interpreter exits after capture).

Reduce helpers
~~~~~~~~~~~~~~
Every sitk.Image is reduced to a stable dict (so changes to default fields do not
cause false failures on content that was never touched):
  ``_reduce_sitk_image`` → dict with keys array/spacing/origin/direction.

NRRD headers contain numpy arrays; ``_reduce_nrrd_header`` converts them to nested
lists so the ``_NumpyEncoder`` serialises them cleanly as JSON.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import _fixtures as fx
import nrrd
import numpy as np
import SimpleITK as sitk
from _capture_golden import CaptureCase

from imatools.common import itktools

# ---------------------------------------------------------------------------
# Persistent temp directory (module-level; lives for the duration of capture)
# ---------------------------------------------------------------------------

_TMPDIR = tempfile.TemporaryDirectory(prefix="imatools_image_io_capture_")
_TMP = Path(_TMPDIR.name)


# ---------------------------------------------------------------------------
# Reduce helpers
# ---------------------------------------------------------------------------


def _reduce_sitk_image(im: sitk.Image) -> Dict[str, Any]:
    """Reduce a sitk.Image to a JSON-serialisable dict."""
    return {
        "array": sitk.GetArrayFromImage(im).tolist(),
        "spacing": list(im.GetSpacing()),
        "origin": list(im.GetOrigin()),
        "direction": list(im.GetDirection()),
    }


def _reduce_nrrd_header(header) -> Dict[str, Any]:
    """Convert NRRD header to a JSON-serialisable dict.

    numpy arrays inside the header are converted to nested lists;
    other values are passed through as-is (they are already str/float/int/tuple).
    """
    out = {}
    for k, v in header.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Write fixture files to temp dir (once, at module load)
# ---------------------------------------------------------------------------

# A .nii.gz image for load_image / load_image_as_np
_NII_PATH = str(_TMP / "label.nii.gz")
sitk.WriteImage(fx.label_image(), _NII_PATH)

# A .nrrd image for load_nrrd_base / get_nrrd_header / load_nrrd_image
_NRRD_PATH = str(_TMP / "label.nrrd")
sitk.WriteImage(fx.label_image(), _NRRD_PATH)

# A .nrrd with non-axis-aligned directions for fix_header_and_save
# We write a nrrd manually to set space directions to a non-diagonal matrix.
_NRRD_OBLIQUE_PATH = str(_TMP / "label_oblique.nrrd")
_data = sitk.GetArrayFromImage(fx.label_image())
_oblique_header = {
    "type": "uint8",
    "dimension": 3,
    "space": "left-posterior-superior",
    "sizes": list(_data.shape),
    "space directions": np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
    "space origin": np.array([0.0, 0.0, 0.0]),
    "endian": "little",
    "encoding": "raw",
}
nrrd.write(_NRRD_OBLIQUE_PATH, _data, _oblique_header)

# A JSON point file for pointfile_to_image
_POINT_JSON_PATH = str(_TMP / "points.json")
_points_dict = {
    "point_0": [3.0, 3.0, 3.0],
    "point_1": [7.0, 7.0, 7.0],
}
with open(_POINT_JSON_PATH, "w", encoding="utf-8") as _fh:
    json.dump(_points_dict, _fh)


# ---------------------------------------------------------------------------
# Wrapper functions for writer-type cases (temp-write-then-read)
# ---------------------------------------------------------------------------


def _capture_save_image_dir() -> Dict[str, Any]:
    """save_image(image, dir, name=...) → reduced sitk.Image."""
    out_dir = str(_TMP)
    out_name = "saved_label.nii.gz"
    itktools.save_image(fx.label_image(), out_dir, name=out_name, manual_ow=True)
    back = sitk.ReadImage(str(_TMP / out_name))
    return _reduce_sitk_image(back)


def _capture_save_image_path() -> Dict[str, Any]:
    """save_image(image, full_path) → reduced sitk.Image."""
    out_path = str(_TMP / "saved_label_path.nii.gz")
    itktools.save_image(fx.label_image(), out_path, manual_ow=True)
    back = sitk.ReadImage(out_path)
    return _reduce_sitk_image(back)


def _capture_fix_header_and_save() -> Dict[str, Any]:
    """fix_header_and_save(src, dst) → reduced NRRD header of the output."""
    out_path = str(_TMP / "label_fixed.nrrd")
    itktools.fix_header_and_save(_NRRD_OBLIQUE_PATH, out_path)
    _, hdr = nrrd.read(out_path)
    return _reduce_nrrd_header(hdr)


def _capture_convert_to_inr() -> Dict[str, Any]:
    """convert_to_inr → read back the .inr and reduce."""
    out_path = str(_TMP / "label.inr")
    itktools.convert_to_inr(fx.label_image(), out_path)
    back = itktools.convert_from_inr(out_path)
    return _reduce_sitk_image(back)


def _capture_convert_from_inr() -> Dict[str, Any]:
    """convert_from_inr → write .inr then convert_from_inr → reduced sitk.Image."""
    inr_path = str(_TMP / "label_roundtrip.inr")
    itktools.convert_to_inr(fx.label_image(), inr_path)
    back = itktools.convert_from_inr(inr_path)
    return _reduce_sitk_image(back)


def _capture_pointfile_to_image() -> Dict[str, Any]:
    """pointfile_to_image(image_path, json_path) → reduced sitk.Image."""
    result = itktools.pointfile_to_image(_NII_PATH, _POINT_JSON_PATH, label=5, girth=1)
    return _reduce_sitk_image(result)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

CASES = [
    # ------------------------------------------------------------------
    # load_image_as_np: returns (array, origin, im_size)
    # ------------------------------------------------------------------
    CaptureCase(
        name="image_io/load_image_as_np_array",
        func=itktools.load_image_as_np,
        args=(_NII_PATH,),
        reduce=lambda result: result[0].tolist(),
        fmt="json",
    ),
    CaptureCase(
        name="image_io/load_image_as_np_meta",
        func=itktools.load_image_as_np,
        args=(_NII_PATH,),
        reduce=lambda result: {
            "origin": list(result[1]),
            "im_size": list(result[2]),
        },
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # load_image: returns sitk.Image
    # ------------------------------------------------------------------
    CaptureCase(
        name="image_io/load_image",
        func=itktools.load_image,
        args=(_NII_PATH,),
        reduce=_reduce_sitk_image,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # load_nrrd_base: returns (data, header)
    # ------------------------------------------------------------------
    CaptureCase(
        name="image_io/load_nrrd_base_array",
        func=itktools.load_nrrd_base,
        args=(_NRRD_PATH,),
        reduce=lambda result: result[0].tolist(),
        fmt="json",
    ),
    CaptureCase(
        name="image_io/load_nrrd_base_header",
        func=itktools.load_nrrd_base,
        args=(_NRRD_PATH,),
        reduce=lambda result: _reduce_nrrd_header(result[1]),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # get_nrrd_header: returns header dict
    # ------------------------------------------------------------------
    CaptureCase(
        name="image_io/get_nrrd_header",
        func=itktools.get_nrrd_header,
        args=(_NRRD_PATH,),
        reduce=_reduce_nrrd_header,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # load_nrrd_image: SKIPPED — master has a bug: it calls
    #   sitk_image.SetSpacing(header['space directions'])
    # passing a (3,3) ndarray where sitk expects a 3-tuple of floats.
    # This crashes on any NRRD written by SimpleITK (which always stores
    # 'space directions' as a matrix, not 'spacings').
    # Captured as an intent-stub test (xfail without golden).
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # save_image (writer: temp-write-then-read)
    # ------------------------------------------------------------------
    CaptureCase(
        name="image_io/save_image_dir",
        func=_capture_save_image_dir,
        args=(),
        fmt="json",
    ),
    CaptureCase(
        name="image_io/save_image_path",
        func=_capture_save_image_path,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # fix_header_and_save (writer: temp-write-then-read)
    # ------------------------------------------------------------------
    CaptureCase(
        name="image_io/fix_header_and_save",
        func=_capture_fix_header_and_save,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # convert_to_inr (writer: write .inr, read back via convert_from_inr)
    # ------------------------------------------------------------------
    CaptureCase(
        name="image_io/convert_to_inr",
        func=_capture_convert_to_inr,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # convert_from_inr (reader: write .inr first, then read)
    # ------------------------------------------------------------------
    CaptureCase(
        name="image_io/convert_from_inr",
        func=_capture_convert_from_inr,
        args=(),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # pointfile_to_image (reader: write image + JSON points, then call)
    # ------------------------------------------------------------------
    CaptureCase(
        name="image_io/pointfile_to_image",
        func=_capture_pointfile_to_image,
        args=(),
        fmt="json",
    ),
]
