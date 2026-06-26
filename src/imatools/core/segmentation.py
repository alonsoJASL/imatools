"""Composed, pure segmentation-editing workflows (sitk.Image -> sitk.Image, no I/O)."""

from typing import Tuple

import SimpleITK as sitk

from imatools.core import image as core_image
from imatools.core import label as core_label


def morph_label(image, label, operation, radius=3, kernel="ball") -> sitk.Image:
    """Apply a morphological op to ONE label and mask the result back into `image`.

    Replaces the duplicated `label_morph` in label_morph.py and multilabel_segmt_tools.py.
    """
    label_im = core_label.extract_single_label(image, label, binarise=True)
    morph_im = core_image.morph_operations(label_im, operation, radius=radius, kernel_type=kernel)
    return core_image.simple_mask(image, morph_im, mask_value=label)


def morph_label_chain(image, labels, operations, radii, kernel="ball") -> sitk.Image:
    """Apply a sequence of single-label morphological ops, each masked back into the image.

    `labels`, `operations`, `radii` are parallel lists (one entry per step); each step
    builds on the previous result. Replaces the temp-file `chain` mode of label_morph.py
    (done in-memory here, since `morph_label` is pure).
    """
    out = image
    for label, operation, radius in zip(labels, operations, radii):
        out = morph_label(out, label, operation, radius=radius, kernel=kernel)
    return out


def compare_label_maps(
    image1: sitk.Image, image2: sitk.Image, im2_swap_axes: bool = False
) -> Tuple[dict, sitk.Image]:
    """Compare two label maps.

    Returns:
      - scores: dict mapping each common label to its Dice score.
      - comparison_image: ``core_label.multilabel_comparison`` over the common labels.
    """
    if image1.GetSize() != image2.GetSize():
        raise ValueError("Images must have the same size to compare.")

    if im2_swap_axes:
        image2 = core_image.swap_axes(image2, [0, 2])  # common axis convention swap

    scores, _ = core_label.compare_images(image1, image2)
    common = list(scores.keys())

    comparison_image = core_label.multilabel_comparison(image1, image2, l1=common, l2=common)

    return scores, comparison_image


def connected_components(image, label) -> Tuple[sitk.Image, list, int]:
    """Connected components of ONE label.

    Returns ``(cc_image, region_labels, num_regions)`` — the label is binarised then passed
    through ``core_label.bwlabeln`` (components relabelled largest-first).
    """
    label_mask = core_label.extract_single_label(image, label, binarise=True)
    return core_label.bwlabeln(label_mask)


def ignore_small_labels(image, labels, min_voxel_size) -> sitk.Image:
    """Zero out any of `labels` whose voxel count is below `min_voxel_size`.

    Renamed from label_connectivity.ignore_labels_with_voxel_size_less_than.
    """
    to_ignore = [ll for ll in labels if core_image.get_num_nonzero_voxels(image == ll) < min_voxel_size]
    return core_label.exchange_many_labels(image, to_ignore, [0] * len(to_ignore))
