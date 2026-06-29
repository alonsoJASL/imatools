"""Image / voxel-array operations migrated from ``imatools.common.itktools`` (T2a2).

The 22 public functions and the ``SegmentationGenerator`` class here are the
authoritative implementations; ``imatools.common.itktools`` and
``imatools.common.SegmentationGenerator`` re-export them via shims at their bottoms.

Helpers that remain in ``itktools`` for now (``imview``,
``get_mask_array_with_restrictions``, ``get_scarq_boundaries``, etc.) are
imported lazily inside the functions that need them, via the ``_itk()`` accessor.
This avoids the circular-import problem: ``itktools`` must finish defining its own
helpers before its bottom shim imports this module.
"""

import numpy as np
import SimpleITK as sitk  # noqa: N813

from imatools.common.config import configure_logging

logger = configure_logging(log_name=__name__)


# ---------------------------------------------------------------------------
# Lazy-helper accessor — avoids circular import at module load time.
# After itktools finishes loading (including its bottom shim), all helper
# names are available in sys.modules and these lookups resolve instantly.
# ---------------------------------------------------------------------------
def _itk():
    """Return the itktools module (always already loaded when an image fn is called)."""
    import imatools.common.itktools as _m  # noqa: PLC0415

    return _m


# ---------------------------------------------------------------------------
# Module-level dictionaries (used by morph_operations)
# ---------------------------------------------------------------------------

MORPH_SWITCHER = {
    "dilate": sitk.BinaryDilate,
    "erode": sitk.BinaryErode,
    "open": sitk.BinaryMorphologicalOpening,
    "close": sitk.BinaryMorphologicalClosing,
    "fill": sitk.BinaryFillhole,
    "smooth": sitk.DiscreteGaussian,  # For smoothing, we'll use DiscreteGaussian.
}

KERNEL_SWITCHER = {"ball": sitk.sitkBall, "box": sitk.sitkBox, "cross": sitk.sitkCross}


# ---------------------------------------------------------------------------
# Image / voxel-array operations — 22 functions (verbatim from master itktools)
# ---------------------------------------------------------------------------


def imarray(im: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(im)


def imview(im: sitk.Image) -> np.ndarray:
    return sitk.GetArrayViewFromImage(im)


def array2im(im_array: np.ndarray, im: sitk.Image) -> sitk.Image:
    """
    Convert a NumPy array to a SimpleITK image.
    """
    im_out = sitk.GetImageFromArray(im_array)
    im_out.CopyInformation(im)

    return im_out


def zeros_like(image):
    """
    Returns a new image with the same size and spacing as the input image, but filled with zeros.
    """
    return sitk.Image(image.GetSize(), sitk.sitkUInt8)


def cp_image(input_image):
    """Copy an image"""
    input_array = imarray(input_image)

    new_image = sitk.GetImageFromArray(input_array)
    new_image.CopyInformation(input_image)

    return new_image


def get_spacing(image: sitk.Image) -> tuple:
    """
    Returns the spacing of the image in mm.
    """
    return image.GetSpacing()


def get_num_nonzero_voxels(image: sitk.Image) -> int:
    """
    Calculates the volume of a binary object in an image.
    """
    image_array = imarray(image)

    # Calculate the volume in mm^3
    return np.count_nonzero(image_array)


def morph_operations(image, operation: str, radius=3, kernel_type="ball"):
    """
    Performs a morphological operation on a binary image with a binary ball of a given radius.
    Additionally, supports Gaussian smoothing using the 'smooth' operation.
    """
    import SimpleITK as sitk  # Ensure SimpleITK is imported  # noqa: N813,PLC0415,F401

    # Define operations for binary morphology and smoothing.
    switcher_operation = MORPH_SWITCHER.copy()
    switcher_kernel = KERNEL_SWITCHER.copy()

    logger.info(
        f"Performing {operation} operation with radius {radius} and kernel type {kernel_type}"
    )

    which_operation = switcher_operation.get(operation, None)
    if which_operation is None:
        raise ValueError(f"Invalid operation: {operation}")

    # For fill, the filter only takes the image.
    if operation == "fill":
        return which_operation(image)

    # If smoothing is requested, apply Gaussian smoothing.
    if operation == "smooth":
        # DiscreteGaussian expects a parameter 'variance'. You can convert your "radius" to variance
        # Here, we use radius directly as variance, but you might adjust this conversion.
        return smooth_label_with_distance(image, sigma=1.0, threshold=0.0)

    # For the binary morphological operations, obtain the kernel type.
    which_kernel = switcher_kernel.get(kernel_type, None)
    if which_kernel is None:
        raise ValueError(f"Invalid kernel type: {kernel_type}")

    # Perform the morphological operation with the specified kernel and radius.
    return which_operation(image, kernelRadius=(radius, radius, radius), kernelType=which_kernel)


def smooth_label_with_distance(image, sigma=1.0, threshold=0.0):
    """
    Smooths a binary label image by converting it to a signed distance map,
    applying Gaussian smoothing, and then re-thresholding.

    Parameters:
      image    : Input binary label image (SimpleITK Image).
      sigma    : Standard deviation for Gaussian smoothing.
      threshold: Threshold value to re-binarize the smoothed distance map.
                 Typically 0.0 works if inside is positive.

    Returns:
      A smoothed binary label image.
    """
    # Convert the label to a signed distance map.
    distance_map = sitk.DanielssonDistanceMap(
        image, inputIsBinary=True, squaredDistance=False, useImageSpacing=True
    )
    # distance_map = sitk.SignedMaurerDistanceMap(image, insideIsPositive=True, useImageSpacing=True)
    # save_image(distance_map, 'distance_map.nrrd')

    # Smooth the distance map.
    smoothed_distance = sitk.DiscreteGaussian(distance_map, variance=sigma**2)
    # save_image(smoothed_distance, 'smoothed_distance.nrrd')

    # Threshold the smoothed distance map to recover a binary image.
    smoothed_label = sitk.BinaryThreshold(
        smoothed_distance,
        lowerThreshold=-1e9,
        upperThreshold=threshold,
        insideValue=1,
        outsideValue=0,
    )

    return smoothed_label


def smooth_labels(im: sitk.Image, sigma=1.0, threshold=0.5, im_close=True):
    unique_labels = _itk().get_labels(im)
    im_size = im.GetSize()

    pixel_type = sitk.sitkUInt8
    output_im = sitk.Image(im_size, pixel_type)
    output_im.CopyInformation(im)

    for label in unique_labels:
        # Create a binary image for the current label
        binary_im = sitk.BinaryThreshold(im, lowerThreshold=label, upperThreshold=label)
        binary_im.CopyInformation(im)

        smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        smooth_filter.SetSigma(sigma)
        label_im = smooth_filter.Execute(binary_im)

        label_im = sitk.BinaryThreshold(label_im, lowerThreshold=threshold, upperThreshold=2)

        if im_close:
            label_im = morph_operations(label_im, "close")

        # Check for overlapping voxels and remove them from label_im
        overlapping_voxels = sitk.And(output_im, sitk.Cast(label_im, pixel_type))
        label_im = sitk.Subtract(sitk.Cast(label_im, pixel_type), overlapping_voxels)
        label_im = sitk.Multiply(label_im, label)

        # Add the resampled label image to the final result
        output_im = sitk.Add(output_im, label_im)

    return output_im


def resample_smooth_label(im: sitk.Image, spacing: list, sigma=3.0, threshold=0.5, im_close=True):
    # import itk

    # Get all unique labels in the image
    unique_labels = _itk().get_labels(im)
    im_size = im.GetSize()
    new_size = [int(im_size[i] * im.GetSpacing()[i] / spacing[i]) for i in range(3)]

    pixel_type = im.GetPixelID()

    # Initialize an empty image to hold the final result
    resampled_im = sitk.Image(new_size, pixel_type)
    resampled_im.SetSpacing(spacing)
    resampled_im.SetOrigin(im.GetOrigin())

    # Resample each label separately
    for label in unique_labels:
        # Create a binary image for the current label
        print(f"Resampling label {label}")

        binary_im = sitk.BinaryThreshold(im, lowerThreshold=label, upperThreshold=label)
        # binary_im = extract_single_label(im, label, binarise=True)

        # Resample the binary image using a Gaussian interpolator
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(spacing)
        resampler.SetSize(resampled_im.GetSize())
        resampler.SetOutputDirection(im.GetDirection())
        resampler.SetOutputOrigin(im.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        # resampler.SetInterpolator(sitk.sitkGaussian)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        resampled_label_im = resampler.Execute(binary_im)
        resampled_label_im.CopyInformation(resampled_im)

        smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        smooth_filter.SetSigma(sigma)
        resampled_label_im = smooth_filter.Execute(resampled_label_im)

        resampled_label_im = sitk.BinaryThreshold(
            resampled_label_im, lowerThreshold=threshold, upperThreshold=2
        )

        if im_close:
            resampled_label_im = morph_operations(resampled_label_im, "close")

        # Check for overlapping voxels and remove them from resampled_label_im
        overlapping_voxels = sitk.And(resampled_im, sitk.Cast(resampled_label_im, pixel_type))
        resampled_label_im = sitk.Subtract(
            sitk.Cast(resampled_label_im, pixel_type), overlapping_voxels
        )
        resampled_label_im = sitk.Multiply(resampled_label_im, label)

        # Add the resampled label image to the final result
        resampled_im = sitk.Add(resampled_im, resampled_label_im)

    return resampled_im


def image_operation(operation, image1, image2=None):
    switcher_operation = {
        "add": sitk.Add,
        "subtract": sitk.Subtract,
        "multiply": sitk.Multiply,
        "divide": sitk.Divide,
        "and": sitk.And,
        "or": sitk.Or,
        "xor": sitk.Xor,
        "not": sitk.Not,
    }

    if image2 is None:
        res_im = switcher_operation.get(operation, lambda: "Invalid operation")(image1)
    else:
        image1 = sitk.Cast(image1, sitk.sitkUInt16)
        image2 = sitk.Cast(image2, sitk.sitkUInt16)
        # save image 1's origin and spacing
        image1_origin = image1.GetOrigin()
        image1_spacing = image1.GetSpacing()
        image1_direction = image1.GetDirection()

        # remove origin from both images
        image1.SetOrigin((0, 0, 0))
        image2.SetOrigin((0, 0, 0))
        res_im = switcher_operation.get(operation, lambda: "Invalid operation")(image1, image2)

        # set the origin and spacing back to image 1's
        res_im.SetOrigin(image1_origin)
        res_im.SetSpacing(image1_spacing)
        res_im.SetDirection(image1_direction)

    res_im.CopyInformation(image1)
    return res_im


def add_images(im1, im2):
    add_im = sitk.Add(im1, im2)
    add_im.CopyInformation(im1)

    return add_im


def simple_mask(im, mask, mask_value=0) -> sitk.Image:
    logger.info(f"Masking image with mask value {mask_value}")
    masked_im_array = imarray(im)
    mask_array = imview(mask)

    masked_im_array[mask_array > 0] = mask_value

    new_im = sitk.GetImageFromArray(masked_im_array)
    new_im.CopyInformation(im)

    return new_im


def mask_image(im, mask, mask_value=0, ignore_im=None, threshold=0):
    masked_im_array = imarray(im)
    mask_array = _itk().get_mask_array_with_restrictions(
        im, mask, threshold=threshold, ignore_im=ignore_im
    )

    masked_im_array[mask_array > 0] = mask_value

    new_im = sitk.GetImageFromArray(masked_im_array)
    new_im.CopyInformation(im)

    return new_im


def get_mask_with_restrictions(im, mask, threshold=0, ignore_im=None) -> sitk.Image:
    """Return a binary mask image with voxel-restriction logic applied.

    Migrated from ``imatools.common.itktools`` (straggler, M1.6a).

    The output mask is 1 where ``mask > 0``, additionally set to 1 where
    ``im > threshold`` (if ``threshold > 0``), and then set to 0 where
    ``ignore_im > 0`` (if ``ignore_im`` is not None).

    Args:
        im:         SimpleITK image used to apply the intensity threshold.
        mask:       Binary mask image (initial foreground).
        threshold:  Intensity threshold above which image voxels are added to mask.
        ignore_im:  Optional image; voxels > 0 here are removed from the mask.

    Returns:
        SimpleITK binary mask image.
    """
    mask_array = _itk().get_mask_array_with_restrictions(
        im, mask, threshold=threshold, ignore_im=ignore_im
    )
    new_mask = sitk.GetImageFromArray(mask_array)
    new_mask.CopyInformation(im)
    return new_mask


def swap_axes(im: sitk.Image, axes: list) -> sitk.Image:
    """
    Swaps the axes of a 3D image according to the given list.
    """
    # Get the image data as a NumPy array
    data = imarray(im)

    # Swap the axes of the NumPy array
    data = np.swapaxes(data, axes[0], axes[1])
    # Create a new SimpleITK image from the modified NumPy array
    new_im = sitk.GetImageFromArray(data)
    new_im.CopyInformation(im)

    return new_im


def regionprops(image: sitk.Image, label=None):
    """
    Returns region properties of a label in the image
    """
    if label is not None:
        image = _itk().extract_single_label(image, label, binarise=True)

    cc_image = sitk.ConnectedComponent(image)
    label_image = sitk.RelabelComponent(cc_image, sortByObjectSize=True)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(label_image)

    return stats


def segmentation_curvature(im: sitk.Image, gradient_sigma=1.0) -> sitk.Image:
    """
    Calculate the segmentation curvature of an input image.

    Parameters:
        im (sitk.Image): The input image.
        gradient_sigma (float): The sigma value for the gradient magnitude filter. Default is 1.0.

    Returns:
        sitk.Image: The segmentation curvature image.
    """
    gradient_filter = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradient_filter.SetSigma(gradient_sigma)
    gradient_im = gradient_filter.Execute(im)

    return gradient_im


def segmentation_curvature_value(im: sitk.Image, gradient_sigma=1.0) -> float:
    """
    Calculate the segmentation curvature of an input image.

    Parameters:
        im (sitk.Image): The input image.
        gradient_sigma (float): The sigma value for the gradient magnitude filter. Default is 1.0.

    Returns:
        float: The segmentation curvature value, which is the mean gradient magnitude divided by the total number of voxels in the image.
    """
    gradient_im = segmentation_curvature(im, gradient_sigma)
    total_voxels = im.GetNumberOfPixels()
    mean_gradient = imview(gradient_im).mean()

    return mean_gradient / total_voxels


def extract_largest(im: sitk.Image) -> sitk.Image:
    """
    Extract the largest connected component from a multilabel image.
    """
    itk = _itk()
    image_labels = itk.get_labels(im)  # noqa: F841
    im_binary = itk.binarise(im)
    cc = sitk.ConnectedComponent(im_binary)
    labels = itk.get_labels(cc)

    im_pixel_type = im.GetPixelID()

    if len(labels) == 1:
        return im

    # Get the size of each connected component
    sizes = []
    for label in labels:
        sizes.append(np.sum(imarray(cc) == label))

    # Find the label with the largest size
    largest_label = labels[np.argmax(sizes)]
    largest_cc = sitk.BinaryThreshold(
        cc, lowerThreshold=largest_label, upperThreshold=largest_label
    )

    # cast the largest connected component to the same pixel type as the input image
    largest_cc = sitk.Cast(largest_cc, im_pixel_type)

    return sitk.Multiply(im, largest_cc)


def generate_scar_image(
    image_size=(300, 300, 100),
    prism_size=(80, 80, 80),
    origin=(0, 0, 0),
    spacing=(1.0, 1.0, 1.0),
    mode="iir",
    simple=False,
    mean_bp_theory=100,
    std_bp_theory=10,
):
    """
    Creates an 'LGE image with scar' for testing purposes.
    """
    print(
        f"Generating {'simple' if simple else 'realistic'} image of size {image_size} with prism of size {prism_size} using method {mode}"
    )

    # Create an image with user-defined dimensions, origin, and spacing
    size_adjusted = (image_size[2], image_size[1], image_size[0])
    image = sitk.Image(size_adjusted, sitk.sitkInt32)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)

    # Set values inside the prism
    start_indx = [(image_size[i] - prism_size[i]) // 2 for i in range(3)]
    end_indx = [start_indx[i] + prism_size[i] for i in range(3)]

    std_bp_theory = std_bp_theory if not simple else 0
    thres = lambda x, mbp, stdb: (  # noqa: E731
        (x * mbp) if mode.lower() == "iir" else (x * stdb + mbp)
    )

    random_background = np.random.randint(0, 20, size=size_adjusted, dtype=np.int32)
    values_inside_prism = np.random.normal(mean_bp_theory, std_bp_theory, size=prism_size).astype(
        np.int32
    )
    mean_bp = np.mean(values_inside_prism)
    std_bp = np.std(values_inside_prism)

    # Create an array with all voxels set to 100 initially
    voxel_values = imarray(image)
    print(f"image size: {image_size}")
    print(f"size_adjusted: {size_adjusted}")
    print(f"size random_background: {random_background.shape}")
    print(f"voxel vlues size: {voxel_values.shape}")

    voxel_values[0 : image_size[2], 0 : image_size[1], 0 : image_size[0]] = random_background
    voxel_values[
        start_indx[0] : end_indx[0], start_indx[1] : end_indx[1], start_indx[2] : end_indx[2]
    ] = values_inside_prism

    # Create a prism mask with the specified dimensions
    prism_mask = imarray(zeros_like(image))
    prism_mask[
        start_indx[0] : end_indx[0], start_indx[1] : end_indx[1], start_indx[2] : end_indx[2]
    ] = 1

    # Create the boundary region of the prism
    boundary_mask = imarray(zeros_like(image))
    boundary_mask[
        start_indx[0] - 1 : end_indx[0] + 1,
        start_indx[1] - 1 : end_indx[1] + 1,
        start_indx[2] - 1 : end_indx[2] + 1,
    ] = 1
    boundary_mask *= 1 - prism_mask  # Exclude the inside of the prism

    total_boundary_mask = np.sum(boundary_mask)

    d = _itk().get_scarq_boundaries(mode)
    percentages = [0.99, 0.01] if simple else [0.6, 0.2, 0.15, 0.05]
    boundic = {
        int(100 * perc): np.round(perc * total_boundary_mask).astype(np.int32)
        for perc in percentages
    }
    totals = [np.round(perc * total_boundary_mask).astype(np.int32) for perc in percentages]

    boundic["mean_bp"] = mean_bp
    boundic["std_bp"] = std_bp

    indices = np.argwhere(boundary_mask == 1)
    np.random.shuffle(indices)
    tx = 0
    for idx in indices:
        try:
            _ = f"Assigning value to voxel {idx} with total {totals[tx]}"
        except IndexError:
            break

        if simple:
            val = 1.05 if tx == 0 else 1.21
            assign_value = thres(val, mean_bp, std_bp).astype(np.int32)
        else:
            low_bound = thres(d[tx][0], mean_bp, std_bp).astype(np.int32)
            high_bound = thres(d[tx][1], mean_bp, std_bp).astype(np.int32)
            assign_value = np.random.randint(low_bound, high_bound, dtype=np.int32)

        voxel_values[idx[0], idx[1], idx[2]] = assign_value
        totals[tx] -= 1

        if totals[tx] == 0:
            tx += 1

    # Set pixel values in the SimpleITK image
    out_image = sitk.GetImageFromArray(voxel_values)
    out_image.CopyInformation(image)

    boundic["total"] = np.sum(list(boundic.values()), dtype=np.int32)
    for key in boundic:
        if isinstance(boundic[key], np.int32):
            boundic[key] = int(boundic[key])

    logger.info(f"Boundaries: {boundic}")

    # Create a segmentation image
    segmentation = sitk.GetImageFromArray(prism_mask)
    segmentation.CopyInformation(image)

    return out_image, segmentation, boundic


def points_to_image(image, points, label=1, girth=2, points_are_indices=False):
    """
    Set the closest voxels to the given points to the specified label in the input image.

    Args:
        image (SimpleITK.Image): The input 3D image.
        points (list of tuples): List of (x, y, z) coordinates of points.
        label (int): The label value to set for the closest voxels to the points.

    Returns:
        SimpleITK.Image: The modified image with the closest voxels set to the label.
    """
    # Convert the input image to a numpy array for easier manipulation
    image_np = imarray(zeros_like(image))
    print(f"Image shape: {image_np.shape}")

    # Loop through each point and find the closest voxel in the image
    for point in points:
        x, y, z = point
        # Convert world coordinates to image indices (pixel coordinates)
        if not points_are_indices:
            print(f"Point: {point}")
            index = image.TransformPhysicalPointToIndex((x, y, z))
            print(f"Index: {index}")
            index_rounded = np.round(index).astype(int)
            print(f"Rounded index: {index_rounded}")
        else:
            index_rounded = np.round(point).astype(int)

        # Set the label for the closest voxel
        for xi in range(-girth, girth):
            for yi in range(-girth, girth):
                for zi in range(-girth, girth):
                    image_np[
                        index_rounded[2] + zi, index_rounded[1] + yi, index_rounded[0] + xi
                    ] = label

    # count number of voxels with label
    # print(f"Number of voxels with label {label}: {np.count_nonzero(image_np == label)}")

    # Convert the modified numpy array back to a SimpleITK image
    modified_image = sitk.GetImageFromArray(image_np)
    modified_image.SetOrigin(image.GetOrigin())
    modified_image.SetSpacing(image.GetSpacing())
    modified_image.SetDirection(image.GetDirection())

    return modified_image


def get_indices_from_label(img: sitk.Image, label: int, get_voxel_bbox=False):
    arr = sitk.GetArrayFromImage(img)
    print(f"{np.unique(arr)=}")
    label = np.array(label, dtype=arr.dtype).item()

    mask = arr == label
    print(f"Num matching voxels: {mask.sum()}")
    vox_indices = np.argwhere(mask)

    world_coords = []
    for idx in vox_indices:
        world_coord = img.TransformIndexToPhysicalPoint(tuple(int(x) for x in reversed(idx)))
        world_coords.append(world_coord)

    if get_voxel_bbox:
        bounding_boxes = []
        bounding_boxes_centres = []
        for idx in vox_indices:
            corners = []
            for dz in [0, 1]:
                for dy in [0, 1]:
                    for dx in [0, 1]:
                        offset = np.array([dz, dy, dx])
                        shifted_idx = idx + offset
                        corner = img.TransformIndexToPhysicalPoint(
                            tuple(int(x) for x in reversed(shifted_idx))
                        )
                        corners.append(corner)
            corner_array = np.array(corners)
            bounding_boxes.append(corner_array)
            bounding_boxes_centres.append(np.mean(corner_array, axis=0))

        return (
            vox_indices,
            world_coords,
            {"centres": bounding_boxes_centres, "corners": bounding_boxes},
        )

    return vox_indices, world_coords


def find_neighbours(image, indices):
    """
    Finds all the unique neighbors and their corresponding pixel values for a list of indices in the given image.

    Args:
        image (SimpleITK.Image): The input image.
        indices (list of tuples): A list of index tuples representing the indices.

    Returns:
        dict: A dictionary with the indices as keys. Each value is a list of tuples containing the neighbor's index
              and its corresponding pixel value.
    """
    logger.info(f"Finding neighbours for {len(indices)} indices.")
    # Get a NumPy array view of the image data
    image_array_view = imview(image)

    # Define the 26-connectivity offsets in 3D space
    offsets = [
        (-1, -1, -1),
        (-1, -1, 0),
        (-1, -1, 1),
        (-1, 0, -1),
        (-1, 0, 0),
        (-1, 0, 1),
        (-1, 1, -1),
        (-1, 1, 0),
        (-1, 1, 1),
        (0, -1, -1),
        (0, -1, 0),
        (0, -1, 1),
        (0, 0, -1),
        (0, 0, 1),
        (0, 1, -1),
        (0, 1, 0),
        (0, 1, 1),
        (1, -1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, 0, -1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1),
    ]

    neighbours_dict = {}

    # Create a set to store the visited indices and initialize it with the input indices
    visited_indices = set(tuple(idx) for idx in indices)

    image_shape = image_array_view.shape
    # Iterate over each index in the list
    for idx in indices:
        x, y, z = idx
        neighbours = []

        # Check all the 26-connectivity neighbours
        for offset in offsets:
            nx, ny, nz = x + offset[0], y + offset[1], z + offset[2]

            # Check if the neighbour is within the image bounds
            if 0 <= nx < image_shape[0] and 0 <= ny < image_shape[1] and 0 <= nz < image_shape[2]:
                neighbour_index = (nx, ny, nz)

                # Check if the neighbour is not already in the visited set
                if neighbour_index not in visited_indices:
                    neighbour_value = image_array_view[nx, ny, nz]
                    # if isinstance(neighbour_value, np.ndarray):
                    #     # If the neighbor's value is an array, convert it to a tuple
                    #     neighbour_value = tuple(neighbour_value.tolist())
                    neighbours.append((neighbour_index, neighbour_value))
                    visited_indices.add(neighbour_index)

        neighbours_dict[idx] = neighbours

    return neighbours_dict


# ---------------------------------------------------------------------------
# SegmentationGenerator class (migrated from imatools.common.SegmentationGenerator)
# Cat-A bug fix: GaussianSource 3rd arg must be a list[float], not a scalar.
# ---------------------------------------------------------------------------


class SegmentationGenerator:
    def __init__(self, size=[300, 300, 100], origin=[0, 0, 0], spacing=[1, 1, 1]):  # noqa: B006
        self.size = size
        self.origin = origin
        self.spacing = spacing

    def generate_circle(self, radius, center):
        image = sitk.Image(self.size, sitk.sitkUInt8)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)

        # Cat-A fix: sigma must be a per-dimension list[float], not a scalar.
        circle = sitk.GaussianSource(
            sitk.sitkUInt8, self.size, [float(radius)] * len(self.size), center
        )
        circle = sitk.Cast(circle, sitk.sitkUInt8)

        image += circle

        return image

    def generate_cube(self, size, origin):
        image = sitk.Image(self.size, sitk.sitkUInt8)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)

        # Cat-A fix: sigma must be a per-dimension list[float], not a scalar.
        cube = sitk.GaussianSource(
            sitk.sitkUInt8, self.size, [float(size)] * len(self.size), origin
        )
        cube = sitk.Cast(cube, sitk.sitkUInt8)

        image += cube

        return image
