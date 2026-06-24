"""Characterization tests for ``imatools.core.label`` (T1a).

All tests import from the TARGET location ``imatools.core.label``.  That module is
currently an empty stub — it will be populated by migration task T2a1.  Until then
every test is marked ``xfail(strict=False)`` so it is collected but does not block CI.

Golden values were captured from master via::

    ~/opt/anaconda3/bin/conda run -n imatools env \\
        PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module label --out tests/golden

where ``M = ~/dev/python/imatools.worktrees/master``.

Master bugs flagged for T2a1
-----------------------------
* ``combine_segmentations`` uses ``BinaryThreshold(upperThreshold=1e9)`` which
  overflows uint8 pixel type and raises "Lower threshold cannot be greater than upper
  threshold."  Golden is captured with uint16 inputs instead.  T2a1 must fix the
  constant to ``255`` or cast inputs inside the function.
* ``distance_based_outlier_detection`` calls ``save_image(distance_map,
  'distance_map.nrrd')`` as a side-effect, writing a file to the current working
  directory.  T2a1 should remove or guard that call.

Comparison helpers
------------------
* **npy** goldens  -> numpy arrays; compared with ``np.testing.assert_array_equal``
  or ``np.testing.assert_allclose``.
* **json** goldens -> Python dicts/scalars/lists; lists are converted to arrays for
  numeric comparison so floating-point serialization round-trips are harmless.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import _fixtures as fx
import numpy as np
import pytest
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_arr(im: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(im)


# ---------------------------------------------------------------------------
# binarise
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_binarise_multilabel(golden):
    from imatools.core.label import binarise

    result = _to_arr(binarise(fx.label_image()))
    expected = golden("label/binarise_multilabel")
    np.testing.assert_array_equal(result, expected)


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_binarise_binary_noop(golden):
    from imatools.core.label import binarise

    result = _to_arr(binarise(fx.binary_image()))
    expected = golden("label/binarise_binary_noop")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# extract_single_label
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_extract_single_label_keep_value(golden):
    from imatools.core.label import extract_single_label

    result = _to_arr(extract_single_label(fx.label_image(), 2, False))
    expected = golden("label/extract_single_label_keep_value")
    np.testing.assert_array_equal(result, expected)


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_extract_single_label_binarise(golden):
    from imatools.core.label import extract_single_label

    result = _to_arr(extract_single_label(fx.label_image(), 2, True))
    expected = golden("label/extract_single_label_binarise")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# merge_label_images
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_merge_label_images(golden):
    from imatools.core.label import merge_label_images

    im0, im1 = fx.label_image_pair()
    result = _to_arr(merge_label_images([im0, im1]))
    expected = golden("label/merge_label_images")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# bwlabeln
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_bwlabeln_binary(golden):
    from imatools.core.label import bwlabeln

    cc_image, labels_list, num_labels = bwlabeln(fx.binary_image())
    expected = golden("label/bwlabeln_binary")
    assert labels_list == expected["labels"]
    assert num_labels == expected["num_labels"]
    np.testing.assert_array_equal(
        np.asarray(_to_arr(cc_image)),
        np.asarray(expected["array"]),
    )


# ---------------------------------------------------------------------------
# relabel_image
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_relabel_image(golden):
    from imatools.core.label import relabel_image

    result = _to_arr(relabel_image(fx.binary_image(), 5))
    expected = golden("label/relabel_image")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# exchange_labels
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_exchange_labels(golden):
    from imatools.core.label import exchange_labels

    result = _to_arr(exchange_labels(fx.label_image(), 1, 10))
    expected = golden("label/exchange_labels")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# get_labels_to_exchange
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_get_labels_to_exchange_no_conflict(golden):
    from imatools.core.label import get_labels_to_exchange

    result = get_labels_to_exchange([1, 2], [10, 20], [1, 2, 3])
    expected = golden("label/get_labels_to_exchange_no_conflict")
    # JSON round-trip: tuples become lists; compare as nested lists
    assert [list(op) for op in result] == expected


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_get_labels_to_exchange_conflict(golden):
    from imatools.core.label import get_labels_to_exchange

    # Swapping 1->2 and 2->3; label 2 is both source and destination (conflict)
    result = get_labels_to_exchange([1, 2], [2, 3], [1, 2, 3])
    expected = golden("label/get_labels_to_exchange_conflict")
    assert [list(op) for op in result] == expected


# ---------------------------------------------------------------------------
# exchange_labels_form_json
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_exchange_labels_form_json(golden):
    from imatools.core.label import exchange_labels_form_json

    old_map = {"a": 1, "b": 2}
    new_map = {"a": 10, "b": 20}

    with tempfile.TemporaryDirectory() as tmpdir:
        old_path = Path(tmpdir) / "old_labels.json"
        new_path = Path(tmpdir) / "new_labels.json"
        old_path.write_text(json.dumps(old_map), encoding="utf-8")
        new_path.write_text(json.dumps(new_map), encoding="utf-8")
        result = _to_arr(exchange_labels_form_json(fx.label_image(), str(old_path), str(new_path)))

    expected = golden("label/exchange_labels_form_json")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# swap_labels
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_swap_labels(golden):
    from imatools.core.label import swap_labels

    result = _to_arr(swap_labels(fx.label_image(), 2, 99))
    expected = golden("label/swap_labels")
    np.testing.assert_array_equal(result, expected)


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_swap_labels_default_new(golden):
    from imatools.core.label import swap_labels

    result = _to_arr(swap_labels(fx.label_image(), 3))
    expected = golden("label/swap_labels_default_new")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# get_labels
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_get_labels_multilabel(golden):
    from imatools.core.label import get_labels

    result = get_labels(fx.label_image())
    expected = golden("label/get_labels_multilabel")
    assert result == expected


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_get_labels_binary(golden):
    from imatools.core.label import get_labels

    result = get_labels(fx.binary_image())
    expected = golden("label/get_labels_binary")
    assert result == expected


# ---------------------------------------------------------------------------
# combine_segmentations
# NOTE: master bug — BinaryThreshold(upperThreshold=1e9) crashes with uint8
# inputs.  Goldens were captured with uint16 inputs; tests use the same cast.
# T2a1 must fix the constant inside the function so uint8 inputs work.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_combine_segmentations_auto_labels(golden):
    from imatools.core.label import combine_segmentations

    _bin_u16 = sitk.Cast(fx.binary_image(), sitk.sitkUInt16)
    result = _to_arr(combine_segmentations([_bin_u16, _bin_u16]))
    expected = golden("label/combine_segmentations_auto_labels")
    np.testing.assert_array_equal(result, expected)


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_combine_segmentations_explicit_labels(golden):
    from imatools.core.label import combine_segmentations

    _bin_u16 = sitk.Cast(fx.binary_image(), sitk.sitkUInt16)
    result = _to_arr(combine_segmentations([_bin_u16, _bin_u16], [5, 6]))
    expected = golden("label/combine_segmentations_explicit_labels")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# gaps
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_gaps_binary(golden):
    from imatools.core.label import gaps

    result = _to_arr(gaps(fx.binary_image()))
    expected = golden("label/gaps_binary")
    np.testing.assert_array_equal(result, expected)


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_gaps_multilabel(golden):
    from imatools.core.label import gaps

    result = _to_arr(gaps(fx.label_image(), multilabel=True))
    expected = golden("label/gaps_multilabel")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# fill_gaps
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_fill_gaps_single(golden):
    from imatools.core.label import fill_gaps

    result = _to_arr(fill_gaps(fx.binary_image()))
    expected = golden("label/fill_gaps_single")
    np.testing.assert_array_equal(result, expected)


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_fill_gaps_two_images(golden):
    from imatools.core.label import fill_gaps

    im0, im1 = fx.label_image_pair()
    result = _to_arr(fill_gaps(im0, im1))
    expected = golden("label/fill_gaps_two_images")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# dice_score
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_dice_score(golden):
    from imatools.core.label import dice_score

    im0, im1 = fx.label_image_pair()
    result = float(dice_score(im0, im1))
    expected = golden("label/dice_score")
    assert result == pytest.approx(float(expected), rel=1e-7)


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_dice_score_identical(golden):
    from imatools.core.label import dice_score

    result = float(dice_score(fx.label_image(), fx.label_image()))
    expected = golden("label/dice_score_identical")
    assert result == pytest.approx(float(expected), rel=1e-7)


# ---------------------------------------------------------------------------
# compare_images
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_compare_images(golden):
    from imatools.core.label import compare_images

    im0, im1 = fx.label_image_pair()
    scores, unique = compare_images(im0, im1)
    expected = golden("label/compare_images")
    assert {str(k): pytest.approx(v, rel=1e-7) for k, v in scores.items()} == {
        k: pytest.approx(v, rel=1e-7) for k, v in expected["scores"].items()
    }
    assert sorted(unique["im1"]) == expected["unique_im1"]
    assert sorted(unique["im2"]) == expected["unique_im2"]


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_compare_images_identical(golden):
    from imatools.core.label import compare_images

    scores, unique = compare_images(fx.label_image(), fx.label_image())
    expected = golden("label/compare_images_identical")
    assert {str(k): pytest.approx(v, rel=1e-7) for k, v in scores.items()} == {
        k: pytest.approx(v, rel=1e-7) for k, v in expected["scores"].items()
    }
    assert sorted(unique["im1"]) == expected["unique_im1"]
    assert sorted(unique["im2"]) == expected["unique_im2"]


# ---------------------------------------------------------------------------
# multilabel_comparison
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_multilabel_comparison(golden):
    from imatools.core.label import multilabel_comparison

    im0, im1 = fx.label_image_pair()
    result = _to_arr(multilabel_comparison(im0, im1))
    expected = golden("label/multilabel_comparison")
    np.testing.assert_array_equal(result, expected)


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_multilabel_comparison_explicit_labels(golden):
    from imatools.core.label import multilabel_comparison

    im0, im1 = fx.label_image_pair()
    result = _to_arr(multilabel_comparison(im0, im1, [1, 2], [1, 2]))
    expected = golden("label/multilabel_comparison_explicit_labels")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# get_labels_volumes
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_get_labels_volumes(golden):
    from imatools.core.label import get_labels_volumes

    result = get_labels_volumes(fx.label_image())
    expected = golden("label/get_labels_volumes")
    # JSON keys are strings; result keys are integers (from LabelStatisticsImageFilter)
    result_str = {str(k): v for k, v in result.items()}
    for key in expected:
        assert result_str[key] == pytest.approx(expected[key], rel=1e-7)


# ---------------------------------------------------------------------------
# distance_based_outlier_detection
# NOTE: master bug — this function writes 'distance_map.nrrd' to the CWD.
# T2a1 must remove or guard that save_image call.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a1", strict=False)
def test_distance_based_outlier_detection(golden, tmp_path, monkeypatch):
    from imatools.core.label import distance_based_outlier_detection

    # Redirect CWD to tmp_path so the side-effect file is written there, not
    # in the repository root.
    monkeypatch.chdir(tmp_path)
    result = _to_arr(distance_based_outlier_detection(fx.label_image(), label=1, gauss_sigma=2.0))
    expected = golden("label/distance_based_outlier_detection")
    np.testing.assert_array_equal(result, expected)
