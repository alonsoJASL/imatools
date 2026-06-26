import os

import nrrd
import numpy as np
import SimpleITK as sitk  # noqa: N813
import vtk

from imatools.common.config import configure_logging

logger = configure_logging(log_name=__name__)


## Tools for header correction orientation
# fix_header_to_axis_aligned moved to core.spatial (T2a3); re-exported via shim below.


# fix_header_and_save, load_image_as_np, load_nrrd_base, load_nrrd_image
# moved to io.image_io (T2a4); re-exported via shim below.
# set_direction_as moved to core.spatial (T2a3); re-exported via shim below.


def load_image(path_to_file, ext="nii"):
    """Reads image into SimpleITK object"""
    logger.info(f"Loading image from {path_to_file}")
    sitk_t1 = sitk.ReadImage(path_to_file)
    return sitk_t1


def get_nrrd_header(path_to_file):
    """
    Reads the NRRD header from a file.
    """
    logger.info(f"Reading NRRD header from {path_to_file}")
    _, header = nrrd.read(path_to_file)

    return header


def explore_labels_to_split(image):
    """
    Returns list of labels that can be split into multiple labels
    """
    labels = get_labels(image)
    labels_to_split = []
    for label in labels:
        _, _, num_cc_labels = bwlabeln(extract_single_label(image, label, binarise=True))
        if num_cc_labels > 1:
            labels_to_split.append(label)

    return labels_to_split


def remove_label(image, label: int):
    """
    Removes a label from a label image.
    """
    image_array = imarray(image)
    image_array[np.equal(image_array, label)] = 0

    new_image = sitk.GetImageFromArray(image_array)
    new_image.CopyInformation(image)

    return new_image


def show_labels(image):
    """
    Prints all the labels in an image.
    """
    labels = get_labels(image)
    print(f"Labels in image: {*labels,}")


# convert_to_inr and convert_from_inr moved to io.image_io (T2a4);
# re-exported via shim below.


def save_image(image, dir_or_path, name=None, manual_ow=False):
    """
    Saves a SimpleITK image to disk.
    """
    output_path = dir_or_path if name is None else os.path.join(dir_or_path, name)
    logger.info(f"Saving image to [{output_path}]")
    if manual_ow and os.path.exists(output_path):
        os.remove(output_path)

    sitk.WriteImage(image, output_path)
    assert os.path.exists(output_path), f"Saving failed! File not found: {output_path}"


# pointfile_to_image moved to io.image_io (T2a4); re-exported via shim below.


def get_mask_array_with_restrictions(im, mask, threshold=0, ignore_im=None) -> np.ndarray:
    mask_array = imarray(mask)
    if threshold > 0:
        im_array = imview(im)
        mask_array[im_array > threshold] = 1

    if ignore_im is not None:
        ignore_im_array = imview(ignore_im)
        mask_array[ignore_im_array > 0] = 0

    mask_array[mask_array > 0] = 1
    return mask_array


def get_mask_with_restrictions(im, mask, threshold=0, ignore_im=None):
    """
    Returns a mask with the following restrictions:
        - mask_value is set to 1
        - mask_value is set to 0 if ignore_im is not None and ignore_im > 0
        - mask_value is set to 0 if mask_value > threshold
    """
    mask_array = get_mask_array_with_restrictions(
        im, mask, threshold=threshold, ignore_im=ignore_im
    )
    new_mask = sitk.GetImageFromArray(mask_array)
    new_mask.CopyInformation(im)

    return new_mask


def check_for_existing_label(im: sitk.Image, label) -> bool:
    """
    Check if a particular label exists in an image
    """
    labels_in_im = get_labels(im)
    return label in labels_in_im


# create_normal_vector_for_plane, create_image_at_plane,
# create_image_at_plane_from_vector moved to core.spatial (T2a3);
# re-exported via shim below.


def project_surface_onto_segmentation(
    segmentation: sitk.Image, surface: vtk.vtkPolyData, check_visited=False
) -> vtk.vtkPolyData:
    import imatools.common.vtktools as vtku  # noqa: PLC0415

    cog = vtku.get_cog_per_element(surface)
    scalars = surface.GetCellData().GetScalars()
    visited_indices = set()
    for ix in range(surface.GetNumberOfCells()):
        x, y, z = cog[ix]
        value = scalars.GetTuple1(ix)
        index = segmentation.TransformPhysicalPointToIndex((x, y, z))

        if visited_indices.__contains__(index) and check_visited:
            continue

        visited_indices.add(index)
        segmentation.SetPixel(index, value)

    return segmentation


def project_segmentation_onto_mesh(
    segmentation: sitk.Image, mesh, check_visited=False
) -> vtk.vtkPolyData:
    import imatools.common.vtktools as vtku  # noqa: PLC0415

    cog = vtku.get_cog_per_element(mesh)
    scalars = mesh.GetCellData().GetScalars()
    visited_indices = set()
    for ix in range(mesh.GetNumberOfCells()):
        x, y, z = cog[ix]
        value = scalars.GetTuple1(ix)
        index = segmentation.TransformPhysicalPointToIndex((x, y, z))

        if visited_indices.__contains__(index) and check_visited:
            continue

        visited_indices.add(index)
        segmentation.SetPixel(index, value)

    return segmentation


def get_scarq_boundaries(mode: str):  #

    iir = mode.lower() == "iir"

    lowthres = 0.9 if iir else 1.1
    fibrosis = 0.975 if iir else 2.0
    scar = 1.21 if iir else 2.2
    ablation = 1.33 if iir else 3.2
    ceiling = 1.5 if iir else 4.0

    bounds = [(lowthres, fibrosis), (fibrosis, scar), (scar, ablation), (ablation, ceiling)]

    return bounds


# ---------------------------------------------------------------------------
# Re-export shims — MUST be at the very bottom of this file so that itktools
# finishes defining its own helpers (imview, get_mask_array_with_restrictions,
# get_scarq_boundaries, …) BEFORE core.label / core.image are imported.
# These bindings make the moved names available in the itktools namespace so
# that any code doing `from imatools.common.itktools import <name>` continues
# to work, and functions defined above that call them resolve correctly.
# ---------------------------------------------------------------------------
from imatools.core.label import (  # noqa: E402,F401,I001
    binarise,
    bwlabeln,
    combine_segmentations,
    compare_images,
    dice_score,
    distance_based_outlier_detection,
    exchange_labels,
    exchange_labels_form_json,
    extract_single_label,
    fill_gaps,
    gaps,
    get_labels,
    get_labels_to_exchange,
    get_labels_volumes,
    merge_label_images,
    multilabel_comparison,
    relabel_image,
    swap_labels,
)
from imatools.core.label import (  # noqa: E402,F401,I001
    exchange_many_labels,
    split_label_into_components as split_labels_on_repeats,  # renamed in migration; legacy name preserved
)
from imatools.core.image import (  # noqa: E402,F401,I001
    add_images,
    array2im,
    cp_image,
    extract_largest,
    find_neighbours,
    generate_scar_image,
    get_indices_from_label,
    get_num_nonzero_voxels,
    get_spacing,
    image_operation,
    imarray,
    imview,
    mask_image,
    morph_operations,
    points_to_image,
    regionprops,
    resample_smooth_label,
    segmentation_curvature,
    segmentation_curvature_value,
    simple_mask,
    smooth_label_with_distance,
    smooth_labels,
    swap_axes,
    zeros_like,
)
from imatools.core.spatial import (  # noqa: E402,F401,I001
    create_image_at_plane,
    create_image_at_plane_from_vector,
    create_normal_vector_for_plane,
    fix_header_to_axis_aligned,
    set_direction_as,
)

# io.image_io shim — file-I/O functions migrated to io/image_io (T2a4).
# Direct re-export (object-identity); the earlier lazy-wrapper workaround for the
# io/__init__ MeshType import crash is no longer needed — contracts now exports MeshType.
from imatools.io.image_io import (  # noqa: E402,F401
    convert_from_inr,
    convert_to_inr,
    fix_header_and_save,
    load_image_as_np,
    load_nrrd_base,
    load_nrrd_image,
    pointfile_to_image,
)
