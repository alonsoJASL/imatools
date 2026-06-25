import os

import SimpleITK as sitk  # noqa: N813
import nrrd
import vtk
import numpy as np
import json

from imatools.common.config import configure_logging

logger = configure_logging(log_name=__name__)


## Tools for header correction orientation
def fix_header_to_axis_aligned(hdr: nrrd.NRRDHeader):
    """Modify NRRD header to make space directions axis-aligned."""
    hdr = hdr.copy()
    dirs = np.asarray(hdr["space directions"], dtype=float)

    # Compute voxel sizes (norms of direction vectors)
    spacings = np.linalg.norm(dirs, axis=1)
    if np.any(spacings <= 0):
        raise ValueError(f"Invalid spacing values: {spacings}")

    # Replace space directions with a diagonal matrix
    aligned_dirs = np.diag(spacings)
    hdr["space directions"] = aligned_dirs

    # Update srow_* fields for ITK/NIfTI compatibility
    origin = hdr["space origin"]
    hdr["srow_x"] = f"{aligned_dirs[0,0]:.6f} 0.000000 0.000000 {origin[0]:.6f}"
    hdr["srow_y"] = f"0.000000 {aligned_dirs[1,1]:.6f} 0.000000 {origin[1]:.6f}"
    hdr["srow_z"] = f"0.000000 0.000000 {aligned_dirs[2,2]:.6f} {origin[2]:.6f}"

    return hdr


def fix_header_and_save(path_to_file, out_path):
    """
    Reads a NRRD file, modifies its header to make space directions axis-aligned,
    and saves the modified header back to a new NRRD file.
    """
    logger.info(f"Fixing header for {path_to_file} and saving to {out_path}")
    data, hdr = nrrd.read(path_to_file)

    # Fix the header
    fixed_header = fix_header_to_axis_aligned(hdr)

    # Save the modified data and header
    nrrd.write(out_path, data, fixed_header)
    logger.info(f"Saved fixed NRRD file to {out_path}")


def set_direction_as(im: sitk.Image, ref: sitk.Image):
    im.SetDirection(ref.GetDirection())
    return im


def load_image_as_np(path_to_file):
    """Reads image into numpy array"""
    sitk_t1 = sitk.ReadImage(path_to_file)

    t1 = imarray(sitk_t1)
    origin = sitk_t1.GetOrigin()
    im_size = sitk_t1.GetSize()

    return t1, origin, im_size


def load_image(path_to_file, ext="nii"):
    """Reads image into SimpleITK object"""
    logger.info(f"Loading image from {path_to_file}")
    sitk_t1 = sitk.ReadImage(path_to_file)
    return sitk_t1


def load_nrrd_base(path_to_file):
    """
    Loads a NRRD file and returns the image and header.
    """
    logger.info(f"Loading NRRD file from {path_to_file}")
    data, header = nrrd.read(path_to_file)

    return data, header


def get_nrrd_header(path_to_file):
    """
    Reads the NRRD header from a file.
    """
    logger.info(f"Reading NRRD header from {path_to_file}")
    _, header = nrrd.read(path_to_file)

    return header


def load_nrrd_image(path_to_file):
    """
    Loads a NRRD file and returns the image as a SimpleITK object.
    """
    logger.info(f"Loading NRRD image from {path_to_file}")
    data, header = load_nrrd_base(path_to_file)

    # Convert the numpy array to a SimpleITK image
    sitk_image = sitk.GetImageFromArray(data)

    # Set the image properties from the header
    sitk_image.SetOrigin(header.get("space origin", (0, 0, 0)))
    sitk_image.SetSpacing(header.get("space directions", (1, 1, 1)))

    return sitk_image


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


def split_labels_on_repeats(image, label: int, open_image=False, open_radius=3):
    """
    Returns new image where label that can be split are split into two distinct
    labels. The largest object gets the original label, while the others get
            label*10 + ix, for ix in range(1, num_splits)
    If any label is already present in image, then that label is 100*label + ix
    """
    forbidden_labels = get_labels(image)
    forbidden_labels.remove(label)

    image_label = extract_single_label(image, label, binarise=True)

    if open_image:
        logger.info(f"Opening image with radius {open_radius}")
        image_label = morph_operations(image_label, "open", radius=open_radius)

    cc_im_label, cc_labels, num_cc_labels = bwlabeln(image_label)
    if num_cc_labels == 1:
        logger.info(f"No connected components found for label {label}")
        return image

    logger.info(f"Found {num_cc_labels} connected components for label {label}")
    image_array = imarray(image)
    image_array[np.equal(image_array, label)] = 0  # remove

    cc_array = imview(cc_im_label)
    for ix, ccl in enumerate(cc_labels):
        new_label = label if ix == 0 else label * 10 + (ccl - 1)

        qx = 1
        while new_label in forbidden_labels:
            new_label = label * np.power(10, qx) + (ccl - 1)
            qx += 1

        image_array[np.equal(cc_array, ccl)] = new_label

    new_image = sitk.GetImageFromArray(image_array)
    new_image.CopyInformation(image)

    return new_image


def show_labels(image):
    """
    Prints all the labels in an image.
    """
    labels = get_labels(image)
    print(f"Labels in image: {*labels,}")


def convert_to_inr(image, out_path):
    """
    Converts a SimpleITK image to an INR file.
    """

    print(f"Converting image to {out_path}")
    # Get the image data as a NumPy array
    data = imview(image)
    spacing = image.GetSpacing()
    # make sure elements less than 1 are 0
    dtype = data.dtype
    bitlen = data.dtype.itemsize * 8
    if dtype == bool or dtype == np.uint8:
        btype = "unsigned fixed"
    elif dtype == np.uint16:
        btype = "unsigned fixed"
    elif dtype == np.int16:
        btype = "signed fixed"
    elif dtype == np.float32:
        btype = "float"
    elif dtype == np.float64:
        btype = "float"
    else:
        raise ValueError("Volume format not supported")

    logger.info(f"Data type: {dtype}. TYPE:{btype} PIXSIZE:{bitlen}")
    xdim, ydim, zdim = data.shape
    header = f"#INRIMAGE-4#{{\nXDIM={xdim}\nYDIM={ydim}\nZDIM={zdim}\nVDIM=1\nVX={spacing[0]:.4f}\nVY={spacing[1]:.4f}\nVZ={spacing[2]:.4f}\n"
    header += "SCALE=2**0\n" if btype == "unsigned fixed" or btype == "signed fixed" else ""
    header += f"TYPE={btype}\nPIXSIZE={bitlen} bits\nCPU=decm"
    header += "\n" * (252 - len(header))  # Fill remaining space with newlines
    header += "##}\n"  # End of header

    # Write to binary file
    with open(out_path, "wb") as file:
        file.write(header.encode(encoding="utf-8"))  # Write header as bytes
        file.write(data.tobytes())  # Write data as bytes


def convert_from_inr(inr_path):
    """
    Converts an INR file to a SimpleITK image.
    """
    with open(inr_path, "rb") as file:
        header = ""
        while True:
            line = file.readline().decode("utf-8")
            header += line
            if line.strip() == "##}":
                break

        # Parse header
        header_dict = {}
        for line in header.split("\n"):
            if "=" in line:
                key, value = line.split("=")
                header_dict[key.strip()] = value.strip()

        xdim = int(header_dict["XDIM"])
        ydim = int(header_dict["YDIM"])
        zdim = int(header_dict["ZDIM"])
        spacing = [float(header_dict["VX"]), float(header_dict["VY"]), float(header_dict["VZ"])]
        pixsize = int(header_dict["PIXSIZE"].split()[0])
        dtype = header_dict["TYPE"]

        if dtype == "unsigned fixed":
            if pixsize == 8:
                np_dtype = np.uint8
            elif pixsize == 16:
                np_dtype = np.uint16
        elif dtype == "signed fixed":
            if pixsize == 16:
                np_dtype = np.int16
        elif dtype == "float":
            if pixsize == 32:
                np_dtype = np.float32
            elif pixsize == 64:
                np_dtype = np.float64
        else:
            raise ValueError("Volume format not supported")

        # Read image data
        data = np.frombuffer(file.read(), dtype=np_dtype)
        data = data.reshape((xdim, ydim, zdim), order="F")  # Fortran order to match INR format

        # Convert to SimpleITK image
        image = sitk.GetImageFromArray(data)
        image.SetSpacing(spacing)

        return image


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


def pointfile_to_image(path_to_image, path_to_points, label=1, girth=2, points_are_indices=False):
    """
    Set the closest voxels to the given points to the specified label in the input image.

    Args:
        path_to_image (str): Path to the input 3D image.
        path_to_points (str): Path to the text file containing the points.
        label (int): The label value to set for the closest voxels to the points.

    Returns:
        SimpleITK.Image: The modified image with the closest voxels set to the label.
    """
    # Load the image
    image = sitk.ReadImage(path_to_image)

    # Load the points
    points = []
    if path_to_points.endswith(".json"):
        with open(path_to_points, "r") as file:
            points_json = json.load(file)
        for _, coordinates in points_json.items():
            points.append(coordinates)

    else:
        with open(path_to_points, "r") as file:
            for line in file:
                # Split the line into a list of strings
                point = line.split()
                # Convert the strings to floats
                point = [float(x) for x in point]
                # Add the point to the list
                points.append(point)

    modified_image = points_to_image(image, points, label, girth, points_are_indices)

    return modified_image


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


def create_normal_vector_for_plane(axis, angle):
    """
    Returns a normal vector for a plane rotated around the given axis by the given angle
    """
    AXES = ["x", "y", "z"]
    if axis not in AXES:
        raise ValueError(f"Axis {axis} not recognised")

    vector = np.zeros(3)
    vector[AXES.index(axis)] = 1

    angle_rad = np.radians(angle)

    if axis == "x":
        rotation_matrix = np.array(
            [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
        )
    elif axis == "y":
        rotation_matrix = np.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        )
    elif axis == "z":
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )

    # Apply the rotation matrix to the initial vector
    normal_vector = np.dot(rotation_matrix, vector)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector


def create_image_at_plane(image: sitk.Image, point_on_plane: np.array, axis: str, angle: float):
    normal_vector = create_normal_vector_for_plane(axis, angle)
    return create_image_at_plane_from_vector(image, point_on_plane, normal_vector)


def create_image_at_plane_from_vector(
    image: sitk.Image, point_on_plane: np.array, normal_vector: np.array
):
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(normal_vector + [0, 0, 1])

    i_transform = transform.GetInverse()

    im_size = image.GetSize()
    spacing = image.GetSpacing()

    # Transform the point on the plane to the image's coordinate system
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([0, 0, -1, 0, -1, 0, 1, 0, 0])
    resampler.SetOutputOrigin(point_on_plane)
    resampler.SetSize(im_size)
    resampler.SetOutputSpacing(spacing)
    resampler.SetTransform(i_transform)

    resampled_im = resampler.Execute(image)

    # Convert the 3D image to a 2D array
    array = imview(resampled_im)

    # Select the middle slice along the third dimension
    slice_index = array.shape[2] // 2
    slice_2d = array[:, :, slice_index]

    return slice_2d


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


def exchange_many_labels(input_image, old_labels: list, new_labels: list):
    labels_in_image = get_labels(input_image)

    swap_ops = get_labels_to_exchange(old_labels, new_labels, labels_in_image)
    new_image = cp_image(input_image)
    for old_label, new_label in swap_ops:
        logger.info(f"Exchanging label {old_label} with {new_label}")
        new_image = exchange_labels(new_image, old_label, new_label)

    return new_image


def imview(im: sitk.Image) -> np.ndarray:
    return sitk.GetArrayViewFromImage(im)


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
