"""Capture cases for the itktools label-algebra set (T1a).

These exercise the functions destined for ``imatools.core.label``.  Each case
records master's exact behaviour as a golden file under ``tests/golden/label/``.

Notes
-----
* ``distance_based_outlier_detection`` calls ``save_image(..., 'distance_map.nrrd')``
  as a side effect (writes to the CWD). The capture still works but will leave a stray
  file in whatever directory the harness is run from. Flagged for T2a1 to fix.

* ``exchange_labels_form_json`` reads two JSON files. We build them inline in a temp
  directory using the standard library, then pass their paths to the function.

* ``bwlabeln`` returns a 3-tuple ``(image, labels_list, num_labels)``.  We reduce to
  ``(array, labels_list, num_labels)`` because the image is characterised separately.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import _fixtures as fx
import SimpleITK as _sitk
from _capture_golden import CaptureCase

from imatools.common import itktools

# ---------------------------------------------------------------------------
# Shared fixture objects (built once at module load; capture env has master code)
# ---------------------------------------------------------------------------

_lbl = fx.label_image()  # multi-label uint8: labels 1, 2, 3
_bin = fx.binary_image()  # binary uint8: label 1 only
_pair = fx.label_image_pair()  # (im0, im1) with partial overlap

# uint16 variant needed for combine_segmentations (see master bug note below)
_bin_u16 = _sitk.Cast(_bin, _sitk.sitkUInt16)


# ---------------------------------------------------------------------------
# Reduce helpers
# ---------------------------------------------------------------------------


def _im_to_arr(im):
    """Reduce a sitk.Image to its voxel array (uint8-safe)."""
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(im)


def _bwlabeln_reduce(result):
    """Return (array, labels_list, num_labels) from bwlabeln's 3-tuple."""
    import SimpleITK as sitk

    cc_image, labels_list, num_labels = result
    return {
        "array": sitk.GetArrayFromImage(cc_image).tolist(),
        "labels": labels_list,
        "num_labels": num_labels,
    }


def _get_labels_volumes_reduce(d):
    """JSON-stringify: cast integer keys to str (JSON object keys are always str)."""
    return {str(k): v for k, v in d.items()}


def _compare_images_reduce(result):
    """compare_images returns (scores_dict, unique_labels_dict)."""
    scores, unique = result
    return {
        "scores": {str(k): float(v) for k, v in scores.items()},
        "unique_im1": sorted(unique["im1"]),
        "unique_im2": sorted(unique["im2"]),
    }


def _multilabel_comparison_reduce(im):
    """Return array from multilabel_comparison result image."""
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(im)


def _distance_outlier_reduce(im):
    """Return array from distance_based_outlier_detection result."""
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(im)


# ---------------------------------------------------------------------------
# exchange_labels_form_json — build temp JSON files inline
# ---------------------------------------------------------------------------


def _exchange_labels_form_json_wrapper():
    """
    Call exchange_labels_form_json with two temp JSON files.
    Swaps label 1 -> 10 and label 2 -> 20 in the label image.
    Returns the result image reduced to a numpy array.
    """
    import SimpleITK as sitk

    old_map = {"a": 1, "b": 2}
    new_map = {"a": 10, "b": 20}

    with tempfile.TemporaryDirectory() as tmpdir:
        old_path = Path(tmpdir) / "old_labels.json"
        new_path = Path(tmpdir) / "new_labels.json"
        old_path.write_text(json.dumps(old_map), encoding="utf-8")
        new_path.write_text(json.dumps(new_map), encoding="utf-8")

        result = itktools.exchange_labels_form_json(fx.label_image(), str(old_path), str(new_path))

    return sitk.GetArrayFromImage(result)


# ---------------------------------------------------------------------------
# CASES
# ---------------------------------------------------------------------------

CASES = [
    # ------------------------------------------------------------------
    # binarise
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/binarise_multilabel",
        func=itktools.binarise,
        args=(_lbl,),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    CaptureCase(
        name="label/binarise_binary_noop",
        func=itktools.binarise,
        args=(_bin,),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # extract_single_label
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/extract_single_label_keep_value",
        func=itktools.extract_single_label,
        args=(_lbl, 2, False),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    CaptureCase(
        name="label/extract_single_label_binarise",
        func=itktools.extract_single_label,
        args=(_lbl, 2, True),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # merge_label_images
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/merge_label_images",
        func=itktools.merge_label_images,
        args=([_pair[0], _pair[1]],),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # bwlabeln
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/bwlabeln_binary",
        func=itktools.bwlabeln,
        args=(_bin,),
        reduce=_bwlabeln_reduce,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # relabel_image
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/relabel_image",
        func=itktools.relabel_image,
        args=(_bin, 5),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # exchange_labels
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/exchange_labels",
        func=itktools.exchange_labels,
        args=(_lbl, 1, 10),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # get_labels_to_exchange
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/get_labels_to_exchange_no_conflict",
        func=itktools.get_labels_to_exchange,
        args=([1, 2], [10, 20], [1, 2, 3]),
        fmt="json",
    ),
    CaptureCase(
        name="label/get_labels_to_exchange_conflict",
        # Swapping 1->2 and 2->3 — label 2 is both src and dst (conflict)
        func=itktools.get_labels_to_exchange,
        args=([1, 2], [2, 3], [1, 2, 3]),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # exchange_labels_form_json
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/exchange_labels_form_json",
        func=_exchange_labels_form_json_wrapper,
        args=(),
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # swap_labels
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/swap_labels",
        func=itktools.swap_labels,
        args=(_lbl, 2, 99),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    CaptureCase(
        name="label/swap_labels_default_new",
        func=itktools.swap_labels,
        args=(_lbl, 3),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # get_labels
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/get_labels_multilabel",
        func=itktools.get_labels,
        args=(_lbl,),
        fmt="json",
    ),
    CaptureCase(
        name="label/get_labels_binary",
        func=itktools.get_labels,
        args=(_bin,),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # combine_segmentations
    # MASTER BUG: combine_segmentations uses BinaryThreshold(lowerThreshold=1,
    # upperThreshold=1e9).  For uint8 pixel type, 1e9 overflows the pixel type
    # and gets clamped to a value < 1, causing ITK to raise "Lower threshold
    # cannot be greater than upper threshold." This affects ALL uint8 inputs.
    # Workaround for capture: cast inputs to uint16 so 1e9 is representable.
    # Flagged for T2a1: fix the constant (e.g. use 255 or cast within the fn).
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/combine_segmentations_auto_labels",
        func=itktools.combine_segmentations,
        args=([_bin_u16, _bin_u16],),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    CaptureCase(
        name="label/combine_segmentations_explicit_labels",
        func=itktools.combine_segmentations,
        args=([_bin_u16, _bin_u16], [5, 6]),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # gaps
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/gaps_binary",
        func=itktools.gaps,
        args=(_bin,),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    CaptureCase(
        name="label/gaps_multilabel",
        func=itktools.gaps,
        args=(_lbl, True),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # fill_gaps
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/fill_gaps_single",
        func=itktools.fill_gaps,
        args=(_bin,),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    CaptureCase(
        name="label/fill_gaps_two_images",
        func=itktools.fill_gaps,
        args=(_pair[0], _pair[1]),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # dice_score — takes sitk.Image; imview used internally
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/dice_score",
        func=itktools.dice_score,
        args=(_pair[0], _pair[1]),
        fmt="json",
    ),
    CaptureCase(
        name="label/dice_score_identical",
        func=itktools.dice_score,
        args=(_lbl, _lbl),
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # compare_images
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/compare_images",
        func=itktools.compare_images,
        args=(_pair[0], _pair[1]),
        reduce=_compare_images_reduce,
        fmt="json",
    ),
    CaptureCase(
        name="label/compare_images_identical",
        func=itktools.compare_images,
        args=(_lbl, _lbl),
        reduce=_compare_images_reduce,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # multilabel_comparison
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/multilabel_comparison",
        func=itktools.multilabel_comparison,
        args=(_pair[0], _pair[1]),
        reduce=_multilabel_comparison_reduce,
        fmt="npy",
    ),
    CaptureCase(
        name="label/multilabel_comparison_explicit_labels",
        func=itktools.multilabel_comparison,
        args=(_pair[0], _pair[1], [1, 2], [1, 2]),
        reduce=_multilabel_comparison_reduce,
        fmt="npy",
    ),
    # ------------------------------------------------------------------
    # get_labels_volumes
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/get_labels_volumes",
        func=itktools.get_labels_volumes,
        args=(_lbl,),
        reduce=_get_labels_volumes_reduce,
        fmt="json",
    ),
    # ------------------------------------------------------------------
    # distance_based_outlier_detection
    # NOTE: this function writes 'distance_map.nrrd' to CWD as a side effect.
    # Flagged for T2a1 to fix. The golden captures the output image array.
    # ------------------------------------------------------------------
    CaptureCase(
        name="label/distance_based_outlier_detection",
        func=itktools.distance_based_outlier_detection,
        args=(_lbl,),
        kwargs={"label": 1, "gauss_sigma": 2.0},
        reduce=_distance_outlier_reduce,
        fmt="npy",
    ),
]
