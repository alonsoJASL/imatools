"""Characterization tests for ``imatools.core.image`` (T1b).

All tests import from the TARGET location ``imatools.core.image``.  That module
is an empty stub — it will be populated by migration task T2a2.  Until then,
every test is marked ``xfail(strict=False)`` so it is collected but does not
block CI.

Two categories of intent-stubs (no golden — master cannot produce one):
1. ``generate_scar_image`` — stochastic (unseeded ``np.random``); flagged for T2a2
   to decide: seed it permanently, or mark as skip-golden.
2. ``SegmentationGenerator.generate_circle`` / ``generate_cube`` — master bug:
   ``sitk.GaussianSource`` 3rd arg must be a *vector* of doubles, but master
   passes a scalar int, causing ``TypeError`` on SITK >= 2.x.  Flagged for T2a2
   to fix the argument before migration lands.

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


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_get_spacing(golden):
    from imatools.core.image import get_spacing

    result = get_spacing(_lbl)
    expected = golden("image/get_spacing")
    assert list(result) == pytest.approx(expected, rel=1e-7)


# ---------------------------------------------------------------------------
# get_num_nonzero_voxels
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_get_num_nonzero_voxels(golden):
    from imatools.core.image import get_num_nonzero_voxels

    result = get_num_nonzero_voxels(_lbl)
    expected = golden("image/get_num_nonzero_voxels")
    assert result == expected


# ---------------------------------------------------------------------------
# zeros_like
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_zeros_like(golden):
    from imatools.core.image import zeros_like

    result = zeros_like(_lbl)
    expected = golden("image/zeros_like")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# cp_image
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_cp_image(golden):
    from imatools.core.image import cp_image

    result = cp_image(_lbl)
    expected = golden("image/cp_image")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# imarray
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_imarray(golden):
    from imatools.core.image import imarray

    result = imarray(_lbl)
    expected = golden("image/imarray")
    np.testing.assert_array_equal(result.flatten().astype(float), expected)


# ---------------------------------------------------------------------------
# array2im
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_array2im(golden):
    from imatools.core.image import array2im

    result = array2im(_ARR, _lbl)
    expected = golden("image/array2im")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# morph_operations — dilate
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_morph_dilate(golden):
    from imatools.core.image import morph_operations

    result = morph_operations(_bin, "dilate", radius=1, kernel_type="ball")
    expected = golden("image/morph_dilate")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# morph_operations — erode
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_morph_erode(golden):
    from imatools.core.image import morph_operations

    result = morph_operations(_bin, "erode", radius=1, kernel_type="ball")
    expected = golden("image/morph_erode")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# morph_operations — close
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_morph_close(golden):
    from imatools.core.image import morph_operations

    result = morph_operations(_bin, "close", radius=1, kernel_type="ball")
    expected = golden("image/morph_close")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# smooth_label_with_distance
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_smooth_label_with_distance(golden):
    from imatools.core.image import smooth_label_with_distance

    result = smooth_label_with_distance(_bin, sigma=1.0, threshold=0.0)
    expected = golden("image/smooth_label_with_distance")
    np.testing.assert_allclose(_im_arr(result), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# smooth_labels
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_smooth_labels(golden):
    from imatools.core.image import smooth_labels

    result = smooth_labels(_lbl, sigma=1.0, threshold=0.5, im_close=True)
    expected = golden("image/smooth_labels")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# resample_smooth_label
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_resample_smooth_label(golden):
    from imatools.core.image import resample_smooth_label

    result = resample_smooth_label(_lbl, _NEW_SPACING, sigma=1.0, threshold=0.5, im_close=True)
    expected = golden("image/resample_smooth_label")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# image_operation — add
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_image_op_add(golden):
    from imatools.core.image import image_operation

    result = image_operation("add", _bin, _bin)
    expected = golden("image/image_op_add")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# image_operation — subtract
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_image_op_subtract(golden):
    from imatools.core.image import image_operation

    result = image_operation("subtract", _bin, _bin)
    expected = golden("image/image_op_subtract")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# image_operation — not (unary)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_image_op_not(golden):
    from imatools.core.image import image_operation

    result = image_operation("not", _bin)
    expected = golden("image/image_op_not")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# add_images
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_add_images(golden):
    from imatools.core.image import add_images

    result = add_images(_bin, _bin)
    expected = golden("image/add_images")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# simple_mask
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_simple_mask(golden):
    from imatools.core.image import simple_mask

    result = simple_mask(_lbl, _bin, mask_value=0)
    expected = golden("image/simple_mask")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# mask_image
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_mask_image(golden):
    from imatools.core.image import mask_image

    result = mask_image(_lbl, _bin, mask_value=0)
    expected = golden("image/mask_image")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# swap_axes
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_swap_axes(golden):
    from imatools.core.image import swap_axes

    result = swap_axes(_lbl, [0, 1])
    expected = golden("image/swap_axes")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# regionprops
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
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


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_segmentation_curvature(golden):
    from imatools.core.image import segmentation_curvature

    result = segmentation_curvature(_bin, gradient_sigma=1.0)
    expected = golden("image/segmentation_curvature")
    np.testing.assert_allclose(_im_arr(result), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# segmentation_curvature_value (scalar)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_segmentation_curvature_value(golden):
    from imatools.core.image import segmentation_curvature_value

    result = segmentation_curvature_value(_bin, gradient_sigma=1.0)
    expected = golden("image/segmentation_curvature_value")
    assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# extract_largest
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_extract_largest(golden):
    from imatools.core.image import extract_largest

    result = extract_largest(_lbl)
    expected = golden("image/extract_largest")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# points_to_image
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_points_to_image(golden):
    from imatools.core.image import points_to_image

    result = points_to_image(_lbl, _POINTS_IDX, label=5, girth=1, points_are_indices=True)
    expected = golden("image/points_to_image")
    np.testing.assert_array_equal(_im_arr(result), expected)


# ---------------------------------------------------------------------------
# get_indices_from_label
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_get_indices_from_label(golden):
    from imatools.core.image import get_indices_from_label

    vox_indices, _world_coords = get_indices_from_label(_lbl, 1, get_voxel_bbox=False)
    expected = golden("image/get_indices_from_label")
    np.testing.assert_array_equal(vox_indices.astype(float), expected)


# ---------------------------------------------------------------------------
# find_neighbours
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
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
# T2a2 must decide: (a) seed before call and lock a golden, or
# (b) keep stochastic and mark this test as skip-golden permanently.
# For now: verify the function exists, can be called, and returns a
# sitk.Image of the expected size.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_generate_scar_image_structural():
    """Intent stub: generate_scar_image returns a sitk.Image of the right size."""
    import SimpleITK as sitk  # noqa: N813

    from imatools.core.image import generate_scar_image

    # Use smallest practical image to keep the test fast.
    image_size = (30, 30, 20)
    prism_size = (10, 10, 10)
    result = generate_scar_image(
        image_size=image_size,
        prism_size=prism_size,
        origin=(0, 0, 0),
        spacing=(1.0, 1.0, 1.0),
        mode="iir",
        simple=True,
    )
    assert isinstance(result, sitk.Image), "generate_scar_image must return a sitk.Image"
    # SITK stores size as (x, y, z); generator sets size_adjusted=(z,y,x)
    assert result.GetSize() == (image_size[2], image_size[1], image_size[0])


# ---------------------------------------------------------------------------
# SegmentationGenerator.generate_circle — INTENT STUB (master bug, no golden)
#
# Master bug: sitk.GaussianSource 3rd arg must be a list of doubles, but
# master's SegmentationGenerator.generate_circle passes a scalar int for
# `radius`, causing TypeError on SITK >= 2.x. No golden can be captured.
# T2a2 must fix: pass [radius]*ndim as a list before calling GaussianSource.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_seggen_generate_circle_structural():
    """Intent stub: SegmentationGenerator.generate_circle returns a sitk.Image."""
    import SimpleITK as sitk  # noqa: N813

    from imatools.core.image import SegmentationGenerator

    gen = SegmentationGenerator(size=[30, 30, 10], origin=[0, 0, 0], spacing=[1, 1, 1])
    result = gen.generate_circle(radius=5, center=[15, 15, 5])
    assert isinstance(result, sitk.Image)
    assert result.GetSize() == (30, 30, 10)


# ---------------------------------------------------------------------------
# SegmentationGenerator.generate_cube — INTENT STUB (master bug, no golden)
#
# Same root cause as generate_circle: scalar passed for `size` to GaussianSource.
# T2a2 must fix: pass [size]*ndim.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="awaiting migration T2a2", strict=False)
def test_seggen_generate_cube_structural():
    """Intent stub: SegmentationGenerator.generate_cube returns a sitk.Image."""
    import SimpleITK as sitk  # noqa: N813

    from imatools.core.image import SegmentationGenerator

    gen = SegmentationGenerator(size=[30, 30, 10], origin=[0, 0, 0], spacing=[1, 1, 1])
    result = gen.generate_cube(size=5, origin=[15, 15, 5])
    assert isinstance(result, sitk.Image)
    assert result.GetSize() == (30, 30, 10)
