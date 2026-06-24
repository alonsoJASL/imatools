"""Capture cases for the ``core/image`` target module (T1b).

Covers the itktools voxel/array ops set and SegmentationGenerator.
``generate_scar_image`` is STOCHASTIC (un-seeded RNG) and is NOT captured;
a structural intent-stub test (xfail) is written in ``test_core_image.py``.

Reduction strategy
------------------
* ``sitk.Image`` -> ``sitk.GetArrayFromImage(result).flatten()`` as npy,
  OR a dict with the array + header metadata for round-trip-sensitive functions.
* ``sitk.LabelShapeStatisticsImageFilter`` (``regionprops``) -> dict of label
  stats extracted via the filter's API.
* ``find_neighbours`` -> JSON; keys are stringified tuples, values are lists
  of ``[list-index, pixel-value]`` pairs.
* ``get_indices_from_label`` (no bbox) -> npy of vox_indices (world_coords
  order is derived, not independently meaningful for the golden contract).
* scalars / tuples / bool -> json.
"""

from __future__ import annotations

import _fixtures as fx
import SimpleITK as sitk
from _capture_golden import CaptureCase

from imatools.common import itktools as itku

# ---------------------------------------------------------------------------
# Shared fixtures (module-level so all cases see the same bytes)
# ---------------------------------------------------------------------------

_lbl = fx.label_image()  # uint8, labels 0/1/2/3, spacing (1,1,2)
_bin = fx.binary_image()  # uint8, binary (0/1), same geometry

# Small points array for points_to_image / find_neighbours
# Using indices directly (points_are_indices=True) to avoid coordinate math.
_POINTS_IDX = [(4, 4, 4), (7, 7, 7)]  # (x, y, z) index triples

# For find_neighbours we need (z, y, x) tuples as indices (numpy axis order).
_NEIGHBOURS_INDICES = [(4, 4, 4), (7, 7, 7)]  # (z, y, x) passed to find_neighbours

# A numpy array to test array2im round-trip
_ARR = itku.imarray(_lbl).copy()  # shape (12,12,12) uint8

# Spacing for resample_smooth_label (coarser than original 1.0x1.0x2.0)
_NEW_SPACING = [2.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# Reducer helpers
# ---------------------------------------------------------------------------


def _im_to_arr(im: sitk.Image):
    """Flatten image array to 1-D for npy storage."""
    return sitk.GetArrayFromImage(im).flatten().astype(float)


def _im_to_dict(im: sitk.Image) -> dict:
    """Serialize image as dict with array + header metadata for json storage."""
    arr = sitk.GetArrayFromImage(im)
    return {
        "array": arr.tolist(),
        "spacing": list(im.GetSpacing()),
        "origin": list(im.GetOrigin()),
        "size": list(im.GetSize()),
    }


def _regionprops_to_dict(stats: sitk.LabelShapeStatisticsImageFilter) -> dict:
    """Extract per-label centroid and voxel count from the filter."""
    labels = sorted(stats.GetLabels())
    result = {}
    for lbl in labels:
        result[str(lbl)] = {
            "voxel_count": stats.GetNumberOfPixels(lbl),
            "centroid": list(stats.GetCentroid(lbl)),
        }
    return result


def _find_neighbours_to_dict(nb_dict: dict) -> dict:
    """Serialize the neighbours dict; keys are stringified tuples."""
    out = {}
    for k, v in nb_dict.items():
        # k is a tuple (z,y,x); stringify for JSON
        str_k = str(k)
        # v is list of ((nz,ny,nx), pixel_value) pairs; store as [list, value]
        out[str_k] = [[list(nb_idx), int(val)] for nb_idx, val in v]
    return out


# ---------------------------------------------------------------------------
# CASES
# ---------------------------------------------------------------------------

CASES = [
    # ----------------------------------------------------------------
    # get_spacing
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/get_spacing",
        func=itku.get_spacing,
        args=(_lbl,),
        reduce=lambda t: list(t),
        fmt="json",
    ),
    # ----------------------------------------------------------------
    # get_num_nonzero_voxels
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/get_num_nonzero_voxels",
        func=itku.get_num_nonzero_voxels,
        args=(_lbl,),
        fmt="json",
    ),
    # ----------------------------------------------------------------
    # zeros_like
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/zeros_like",
        func=itku.zeros_like,
        args=(_lbl,),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # cp_image
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/cp_image",
        func=itku.cp_image,
        args=(_lbl,),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # imarray
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/imarray",
        func=itku.imarray,
        args=(_lbl,),
        reduce=lambda arr: arr.flatten().astype(float),
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # array2im — use the label array + label image as reference
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/array2im",
        func=itku.array2im,
        args=(_ARR, _lbl),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # morph_operations — dilate on binary image
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/morph_dilate",
        func=itku.morph_operations,
        args=(_bin, "dilate"),
        kwargs={"radius": 1, "kernel_type": "ball"},
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # morph_operations — erode on binary image
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/morph_erode",
        func=itku.morph_operations,
        args=(_bin, "erode"),
        kwargs={"radius": 1, "kernel_type": "ball"},
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # morph_operations — close on binary image
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/morph_close",
        func=itku.morph_operations,
        args=(_bin, "close"),
        kwargs={"radius": 1, "kernel_type": "ball"},
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # smooth_label_with_distance
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/smooth_label_with_distance",
        func=itku.smooth_label_with_distance,
        args=(_bin,),
        kwargs={"sigma": 1.0, "threshold": 0.0},
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # smooth_labels (label image)
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/smooth_labels",
        func=itku.smooth_labels,
        args=(_lbl,),
        kwargs={"sigma": 1.0, "threshold": 0.5, "im_close": True},
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # resample_smooth_label
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/resample_smooth_label",
        func=itku.resample_smooth_label,
        args=(_lbl, _NEW_SPACING),
        kwargs={"sigma": 1.0, "threshold": 0.5, "im_close": True},
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # image_operation — add (two images)
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/image_op_add",
        func=itku.image_operation,
        args=("add", _bin, _bin),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # image_operation — subtract (two images)
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/image_op_subtract",
        func=itku.image_operation,
        args=("subtract", _bin, _bin),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # image_operation — not (single image, unary)
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/image_op_not",
        func=itku.image_operation,
        args=("not", _bin),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # add_images
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/add_images",
        func=itku.add_images,
        args=(_bin, _bin),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # simple_mask
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/simple_mask",
        func=itku.simple_mask,
        args=(_lbl, _bin),
        kwargs={"mask_value": 0},
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # mask_image (basic: no ignore_im, no threshold)
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/mask_image",
        func=itku.mask_image,
        args=(_lbl, _bin),
        kwargs={"mask_value": 0},
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # swap_axes — swap axes 0 and 1 (z and y in numpy order)
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/swap_axes",
        func=itku.swap_axes,
        args=(_lbl, [0, 1]),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # regionprops (no label — full image)
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/regionprops",
        func=itku.regionprops,
        args=(_bin,),
        reduce=_regionprops_to_dict,
        fmt="json",
    ),
    # ----------------------------------------------------------------
    # segmentation_curvature (image)
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/segmentation_curvature",
        func=itku.segmentation_curvature,
        args=(_bin,),
        kwargs={"gradient_sigma": 1.0},
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # segmentation_curvature_value (scalar)
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/segmentation_curvature_value",
        func=itku.segmentation_curvature_value,
        args=(_bin,),
        kwargs={"gradient_sigma": 1.0},
        fmt="json",
    ),
    # ----------------------------------------------------------------
    # extract_largest
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/extract_largest",
        func=itku.extract_largest,
        args=(_lbl,),
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # points_to_image (using indices directly)
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/points_to_image",
        func=itku.points_to_image,
        args=(_lbl, _POINTS_IDX),
        kwargs={"label": 5, "girth": 1, "points_are_indices": True},
        reduce=_im_to_arr,
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # get_indices_from_label (no bbox)
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/get_indices_from_label",
        func=itku.get_indices_from_label,
        args=(_lbl, 1),
        kwargs={"get_voxel_bbox": False},
        # Returns (vox_indices, world_coords); capture only vox_indices as npy
        reduce=lambda r: r[0].astype(float),
        fmt="npy",
    ),
    # ----------------------------------------------------------------
    # find_neighbours
    # ----------------------------------------------------------------
    CaptureCase(
        name="image/find_neighbours",
        func=itku.find_neighbours,
        args=(_lbl, _NEIGHBOURS_INDICES),
        reduce=_find_neighbours_to_dict,
        fmt="json",
    ),
    # NOTE: SegmentationGenerator.generate_circle and generate_cube are NOT captured.
    # Master bug: GaussianSource 3rd argument must be a list/vector of doubles, but
    # master passes a scalar (radius=5, size=5) which raises TypeError on SITK>=2.x.
    # Both methods are flagged as intent-stubs in test_core_image.py for T2a2 to fix.
]
