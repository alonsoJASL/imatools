"""Characterization tests for ``imatools.core.image`` (T1b).

All tests import from the TARGET location ``imatools.core.image``.  Migration
task T2a2 has populated this module and removed the ``xfail`` markers so these
tests run against the real implementation.

Two categories of intent-stubs (no golden — master cannot produce one):
1. ``generate_scar_image`` — stochastic (unseeded ``np.random``); structural
   only — verifies the function is importable and returns the right types.
2. ``SegmentationGenerator.generate_circle`` / ``generate_cube`` — Cat-A bug
   fixed in T2a2: ``sitk.GaussianSource`` 3rd arg is now a ``list[float]``
   (was a scalar in master, causing ``TypeError`` on SITK >= 2.x).

Golden values were captured from master via::

    ~/opt/anaconda3/bin/conda run -n imatools env \\
        PYTHONPATH=$M:$M/imatools \\
        python tests/_capture_golden.py --module image --out tests/golden

where ``M = ~/dev/python/imatools.worktrees/master``.
"""

from __future__ import annotations

import _fixtures as fx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixture inputs (module-level — same bytes as capture-time)
# ---------------------------------------------------------------------------

_lbl = fx.label_image()  # uint8, labels 0/1/2/3, spacing (1,1,2)
_bin = fx.binary_image()  # uint8, binary (0/1), same geometry
_ARR = fx.label_array()  # raw numpy array for array2im
_POINTS_IDX = [(4, 4, 4), (7, 7, 7)]
_NEIGHBOURS_INDICES = [(4, 4, 4), (7, 7, 7)]
_NEW_SPACING = [2.0, 2.0, 4.0]

# ---------------------------------------------------------------------------
# Helper: reduce a sitk.Image to a comparable numpy array
# ---------------------------------------------------------------------------


def _im_arr(im):
    import SimpleITK as sitk  # noqa: N813

    return sitk.GetArrayFromImage(im).flatten().astype(float)


# ---------------------------------------------------------------------------
# get_spacing
# ---------------------------------------------------------------------------


def test_get_spacing(golden):
    from imatools.core.image import get_spacing

    result = get_spacing(_lbl)
    expected = golden("image/get_spacing")
    assert list(result) == pytest.approx(expected, rel=1e-7)


# ---------------------------------------------------------------------------
# get_num_nonzero_voxels
# ---------------------------------------------------------------------------


def test_get_num_nonzero_voxels(golden):
    from imatools.core.image import get_num_nonzero_voxels

    result = get_num_nonzero_voxels(_lbl)
    expected = golden("image/get_num_nonzero_voxels")
    assert result == expected


# ---------------------------------------------------------------------------
# zeros_like
# ---------------------------------------------------------------------------


def test_zeros_like(golden):
    from imatools.core.image import zeros_like

    result = zeros_like(_lbl)
    expected = golden("image/zeros_like")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# cp_image
# ---------------------------------------------------------------------------


def test_cp_image(golden):
    from imatools.core.image import cp_image

    result = cp_image(_lbl)
    expected = golden("image/cp_image")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# imarray
# ---------------------------------------------------------------------------


def test_imarray(golden):
    from imatools.core.image import imarray

    result = imarray(_lbl)
    expected = golden("image/imarray")
    np.testing.assert_array_equal(result.flatten().astype(float), expected)


# ---------------------------------------------------------------------------
# array2im
# ---------------------------------------------------------------------------


def test_array2im(golden):
    from imatools.core.image import array2im

    result = array2im(_ARR, _lbl)
    expected = golden("image/array2im")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# morph_operations — dilate
# ---------------------------------------------------------------------------


def test_morph_dilate(golden):
    from imatools.core.image import morph_operations

    result = morph_operations(_bin, "dilate", radius=1, kernel_type="ball")
    expected = golden("image/morph_dilate")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# morph_operations — erode
# ---------------------------------------------------------------------------


def test_morph_erode(golden):
    from imatools.core.image import morph_operations

    result = morph_operations(_bin, "erode", radius=1, kernel_type="ball")
    expected = golden("image/morph_erode")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# morph_operations — close
# ---------------------------------------------------------------------------


def test_morph_close(golden):
    from imatools.core.image import morph_operations

    result = morph_operations(_bin, "close", radius=1, kernel_type="ball")
    expected = golden("image/morph_close")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# smooth_label_with_distance
# ---------------------------------------------------------------------------


def test_smooth_label_with_distance(golden):
    from imatools.core.image import smooth_label_with_distance

    result = smooth_label_with_distance(_bin, sigma=1.0, threshold=0.0)
    expected = golden("image/smooth_label_with_distance")
    np.testing.assert_allclose(_im_arr(result), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# smooth_labels
# ---------------------------------------------------------------------------


def test_smooth_labels(golden):
    from imatools.core.image import smooth_labels

    result = smooth_labels(_lbl, sigma=1.0, threshold=0.5, im_close=True)
    expected = golden("image/smooth_labels")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# resample_smooth_label
# ---------------------------------------------------------------------------


def test_resample_smooth_label(golden):
    from imatools.core.image import resample_smooth_label

    result = resample_smooth_label(_lbl, _NEW_SPACING, sigma=1.0, threshold=0.5, im_close=True)
    expected = golden("image/resample_smooth_label")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# image_operation — add
# ---------------------------------------------------------------------------


def test_image_op_add(golden):
    from imatools.core.image import image_operation

    result = image_operation("add", _bin, _bin)
    expected = golden("image/image_op_add")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# image_operation — subtract
# ---------------------------------------------------------------------------


def test_image_op_subtract(golden):
    from imatools.core.image import image_operation

    result = image_operation("subtract", _bin, _bin)
    expected = golden("image/image_op_subtract")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# image_operation — not (unary)
# ---------------------------------------------------------------------------


def test_image_op_not(golden):
    from imatools.core.image import image_operation

    result = image_operation("not", _bin)
    expected = golden("image/image_op_not")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# add_images
# ---------------------------------------------------------------------------


def test_add_images(golden):
    from imatools.core.image import add_images

    result = add_images(_bin, _bin)
    expected = golden("image/add_images")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# simple_mask
# ---------------------------------------------------------------------------


def test_simple_mask(golden):
    from imatools.core.image import simple_mask

    result = simple_mask(_lbl, _bin, mask_value=0)
    expected = golden("image/simple_mask")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# mask_image
# ---------------------------------------------------------------------------


def test_mask_image(golden):
    from imatools.core.image import mask_image

    result = mask_image(_lbl, _bin, mask_value=0)
    expected = golden("image/mask_image")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# swap_axes
# ---------------------------------------------------------------------------


def test_swap_axes(golden):
    from imatools.core.image import swap_axes

    result = swap_axes(_lbl, [0, 1])
    expected = golden("image/swap_axes")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# regionprops
# ---------------------------------------------------------------------------


def test_regionprops(golden):
    from imatools.core.image import regionprops

    stats = regionprops(_bin)
    expected = golden("image/regionprops")
    # Verify label list and voxel counts match.
    labels = sorted(stats.GetLabels())
    assert [str(lbl) for lbl in labels] == sorted(expected.keys())
    for lbl in labels:
        exp_entry = expected[str(lbl)]
        assert stats.GetNumberOfPixels(lbl) == exp_entry["voxel_count"]


# ---------------------------------------------------------------------------
# segmentation_curvature (image output)
# ---------------------------------------------------------------------------


def test_segmentation_curvature(golden):
    from imatools.core.image import segmentation_curvature

    result = segmentation_curvature(_bin, gradient_sigma=1.0)
    expected = golden("image/segmentation_curvature")
    np.testing.assert_allclose(_im_arr(result), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# segmentation_curvature_value (scalar)
# ---------------------------------------------------------------------------


def test_segmentation_curvature_value(golden):
    from imatools.core.image import segmentation_curvature_value

    result = segmentation_curvature_value(_bin, gradient_sigma=1.0)
    expected = golden("image/segmentation_curvature_value")
    assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# extract_largest
# ---------------------------------------------------------------------------


def test_extract_largest(golden):
    from imatools.core.image import extract_largest

    result = extract_largest(_lbl)
    expected = golden("image/extract_largest")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# points_to_image
# ---------------------------------------------------------------------------


def test_points_to_image(golden):
    from imatools.core.image import points_to_image

    result = points_to_image(_lbl, _POINTS_IDX, label=5, girth=1, points_are_indices=True)
    expected = golden("image/points_to_image")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# get_indices_from_label
# ---------------------------------------------------------------------------


def test_get_indices_from_label(golden):
    from imatools.core.image import get_indices_from_label

    vox_indices, _world_coords = get_indices_from_label(_lbl, 1, get_voxel_bbox=False)
    expected = golden("image/get_indices_from_label")
    np.testing.assert_array_equal(vox_indices.astype(float), expected)


# ---------------------------------------------------------------------------
# find_neighbours
# ---------------------------------------------------------------------------


def test_find_neighbours(golden):
    from imatools.core.image import find_neighbours

    result = find_neighbours(_lbl, _NEIGHBOURS_INDICES)
    expected = golden("image/find_neighbours")
    # Compare key set (stringified tuples) and neighbour counts.
    result_str = {str(k): v for k, v in result.items()}
    assert set(result_str.keys()) == set(expected.keys())
    for k in expected:
        assert len(result_str[k]) == len(expected[k]), f"neighbour count mismatch at {k}"


# ---------------------------------------------------------------------------
# generate_scar_image — INTENT STUB (stochastic, no golden)
#
# Master uses np.random.randint / np.random.normal WITHOUT seeding, so the
# output is non-deterministic. No golden can be captured.
# T2a2 decision: keep stochastic (Cat B — preserve behaviour verbatim).
# Structural test: verify the function is callable and returns (sitk.Image,
# sitk.Image, dict) — the 3-tuple that master always produced.
# ---------------------------------------------------------------------------


def test_generate_scar_image_structural():
    """Intent stub: generate_scar_image returns a 3-tuple (out_image, segmentation, boundic)."""
    import SimpleITK as sitk  # noqa: N813

    from imatools.core.image import generate_scar_image

    # Use a cubic image to avoid the Cat-B master bug where non-cubic dims
    # cause a shape mismatch in the random_background assignment.
    image_size = (30, 30, 30)
    prism_size = (10, 10, 10)
    out_image, segmentation, boundic = generate_scar_image(
        image_size=image_size,
        prism_size=prism_size,
        origin=(0, 0, 0),
        spacing=(1.0, 1.0, 1.0),
        mode="iir",
        simple=True,
    )
    assert isinstance(
        out_image, sitk.Image
    ), "generate_scar_image must return a sitk.Image as first element"
    # SITK stores size as (x, y, z); generator sets size_adjusted=(z,y,x)
    assert out_image.GetSize() == (image_size[2], image_size[1], image_size[0])


# ---------------------------------------------------------------------------
# SegmentationGenerator.generate_circle — INTENT STUB (Cat-A fix applied)
#
# T2a2 fix: sitk.GaussianSource 3rd arg is now [float(radius)]*ndim (list),
# not a scalar. This resolves the TypeError on SITK >= 2.x.
# ---------------------------------------------------------------------------


def test_seggen_generate_circle_structural():
    """Intent stub: SegmentationGenerator.generate_circle returns a sitk.Image."""
    import SimpleITK as sitk  # noqa: N813

    from imatools.core.image import SegmentationGenerator

    gen = SegmentationGenerator(size=[30, 30, 10], origin=[0, 0, 0], spacing=[1, 1, 1])
    result = gen.generate_circle(radius=5, center=[15, 15, 5])
    assert isinstance(result, sitk.Image)
    assert result.GetSize() == (30, 30, 10)


# ---------------------------------------------------------------------------
# SegmentationGenerator.generate_cube — INTENT STUB (Cat-A fix applied)
#
# Same root cause as generate_circle: scalar passed for `size` to GaussianSource.
# T2a2 fix: pass [float(size)]*ndim.
# ---------------------------------------------------------------------------


def test_seggen_generate_cube_structural():
    """Intent stub: SegmentationGenerator.generate_cube returns a sitk.Image."""
    import SimpleITK as sitk  # noqa: N813

    from imatools.core.image import SegmentationGenerator

    gen = SegmentationGenerator(size=[30, 30, 10], origin=[0, 0, 0], spacing=[1, 1, 1])
    result = gen.generate_cube(size=5, origin=[15, 15, 5])
    assert isinstance(result, sitk.Image)
    assert result.GetSize() == (30, 30, 10)
