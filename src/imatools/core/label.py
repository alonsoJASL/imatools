# src/imatools/core/label.py
"""Label-algebra functions migrated from ``imatools.common.itktools`` (T2a1).

The 18 public functions here are the authoritative implementation.

The voxel-array helpers used below (``imarray``, ``imview``, ``array2im``,
``cp_image``, ``morph_operations``, ``image_operation``, ``find_neighbours``)
live in ``core.image``; they are imported lazily via the ``_image()`` accessor
to avoid the label↔image circular import (M2c — was routed through the
``common.itktools`` shim, now gone).
"""

import json

import numpy as np
import SimpleITK as sitk  # noqa: N813

from imatools.common.config import configure_logging

logger = configure_logging(log_name=__name__)


# ---------------------------------------------------------------------------
# Lazy-helper accessor — core.image imports nothing from core.label at load
# time, but keeping this lazy matches the rest of the layer and is robust to
# future edits (M2c — was routed through the ``common.itktools`` shim, now gone).
# ---------------------------------------------------------------------------
def _image():
    """Return the core.image module (imported lazily to avoid the label↔image cycle)."""
    import imatools.core.image as _m  # noqa: PLC0415

    return _m


# ---------------------------------------------------------------------------
# Label algebra — 18 functions (verbatim from master itktools)
# ---------------------------------------------------------------------------


def binarise(image, background=0, foreground=1):
    """Return an image with ``foreground`` where ``image > background``, else 0.

    The default (``background=0, foreground=1``) is a plain 0/1 binarisation.
    The output dtype is the smallest unsigned-int type that holds ``foreground``
    (uint8 for the default), widened as needed so a large ``foreground`` never
    overflows — this is what ``relabel_image`` (now an alias) relies on.
    """
    itk = _image()
    image_array = itk.imarray(image)
    out_dtype = np.promote_types(np.uint8, np.min_scalar_type(int(foreground)))
    bin_array = np.zeros(image_array.shape, dtype=out_dtype)
    bin_array[np.greater(image_array, background)] = foreground
    binim = sitk.GetImageFromArray(bin_array)
    binim.CopyInformation(image)

    return binim


def extract_single_label(image, label, binarise=False) -> sitk.Image:
    """
    Extracts a single label from a label map image.
    """
    itk = _image()
    image_array = itk.imview(image)
    label = np.array(label, dtype=image_array.dtype).item()

    label_array = np.zeros(image_array.shape, dtype=image_array.dtype)
    label_array[np.equal(image_array, label)] = 1 if binarise else label
    label_image = sitk.GetImageFromArray(label_array)
    label_image.CopyInformation(image)
    label_image = sitk.Cast(label_image, sitk.sitkUInt8)
    return label_image


def merge_label_images(images):
    """
    Merges a list of label images into a single label image.
    """
    merged_image = sitk.Image(images[0].GetSize(), sitk.sitkUInt8)
    merged_image.CopyInformation(images[0])
    for image in images:
        merged_image = merged_image + image
    return merged_image


def bwlabeln(image):
    """
    Returns an image where all separated components have a different tag
    """
    binim = binarise(image)
    cc_image = sitk.ConnectedComponent(binim)
    sorted_cc_image = sitk.RelabelComponent(cc_image, sortByObjectSize=True)

    labels = get_labels(sorted_cc_image)
    num_labels = len(labels)

    return sorted_cc_image, labels, num_labels


def relabel_image(input_image, new_label):
    """Deprecated alias for ``binarise(image, background=0, foreground=new_label)``.

    Despite the name this never remapped labels — it thresholds ``> 0`` and sets
    the whole foreground to ``new_label`` (i.e. binarise with a custom foreground
    value). Use ``binarise`` directly; ``exchange_labels`` is the real label remap.
    """
    return binarise(input_image, background=0, foreground=new_label)


def exchange_labels(input_image, old_label, new_label):
    itk = _image()
    input_array = itk.imarray(input_image)

    # Ensure it's an int type with enough bits for your labels
    if not np.issubdtype(input_array.dtype, np.integer):
        input_array = input_array.astype(np.int32)

    old_label = int(old_label)
    new_label = int(new_label)

    # Use np.where to replace old_label with new_label
    input_array = np.where(input_array == old_label, new_label, input_array)

    new_image = sitk.GetImageFromArray(input_array)
    new_image.CopyInformation(input_image)

    return new_image


def get_labels_to_exchange(
    old_labels: list[int], new_labels: list[int], labels_in_image: list[int]
) -> list[tuple[int, int]]:
    """
    Build an ordered list of swap operations (tuples of (source, target))
    so that if a destination label is also a source in another mapping,
    it is first remapped to a temporary label.

    For example, for swapping 4->5 and 5->6, we produce:
      [(5, temp), (4,5), (temp,6)]
    so that the conflicting label 5 is freed before assigning it.

    Parameters:
      old_labels: list of labels to replace.
      new_labels: list of target labels.
      labels_in_image: the list of all labels present in the image.

    Returns:
      A list of (source, target) swap operations in the order they should be applied.
    """
    swap_ops = []
    temp_map = (
        {}
    )  # Map a label that is used as a destination and is also in old_labels to a temporary value.
    additional_label_count = 1
    max_label_value = max(labels_in_image)

    # Phase 1: Identify conflicts and assign temporary labels.
    # We consider it a conflict if a destination label (new) appears in the list of old labels.
    for old, new in zip(old_labels, new_labels):
        if old == new:
            print(f"Old label {old} is the same as new label {new}. Skipping...")
            continue
        if new in old_labels:
            if new not in temp_map:
                temp_label = max_label_value + additional_label_count
                additional_label_count += 1
                temp_map[new] = temp_label
                print(
                    f"Conflict detected: new label {new} appears as a source. Using temporary label {temp_label}."
                )

    # Phase 2: Build the swap sequence.
    # First, remove the conflicting label by mapping it to its temporary value.
    for conflict_label, temp_label in temp_map.items():
        swap_ops.append((conflict_label, temp_label))

    # Then, for each intended mapping:
    # If the source was remapped (i.e. is in temp_map), use the temporary label for the swap.
    for old, new in zip(old_labels, new_labels):
        if old == new:
            continue
        if old in temp_map:
            # Use the temporary label instead of the original source.
            swap_ops.append((temp_map[old], new))
        else:
            swap_ops.append((old, new))

    return swap_ops


def exchange_labels_form_json(input_image, json_old: str, json_new: str):
    with open(json_old, "r") as f:
        old_labels_json = json.load(f)
    with open(json_new, "r") as f:
        new_labels_json = json.load(f)

    old_labels = list(old_labels_json.values())
    new_labels = list(new_labels_json.values())

    return exchange_many_labels(input_image, old_labels, new_labels)


def swap_labels(im, old_label: int, new_label=1):
    """
    Swaps all instances of old_label with new_label in a label image.
    """
    itk = _image()
    im_array = itk.imarray(im)
    im_array[np.equal(im_array, old_label)] = new_label

    new_image = sitk.GetImageFromArray(im_array)

    new_image.SetOrigin(im.GetOrigin())
    new_image.SetSpacing(im.GetSpacing())
    new_image.SetDirection(im.GetDirection())

    return new_image


def get_labels(image: sitk.Image) -> list:
    """
    Returns a list of labels in an image.
    """
    itk = _image()
    image_array_view = itk.imview(image)
    unique_labels = np.unique(image_array_view)
    unique_labels = unique_labels[unique_labels != 0]

    return unique_labels.astype(int).tolist()


def check_for_existing_label(im: sitk.Image, label) -> bool:
    """
    Check if a particular label exists in an image
    """
    labels_in_im = get_labels(im)
    return label in labels_in_im


def combine_segmentations(seg_images, labels=None):
    """
    Combines multiple segmentation images into a single label image.

    Each segmentation image is assumed to be binary (0 for background and non-zero for foreground)
    and to share the same physical geometry. For each segmentation, a distinct label value is assigned.

    Parameters:
      seg_images (list of sitk.Image): List of segmentation images.
      labels (list of int, optional): List of label values. If None, labels will be assigned as 1,2,3,...

    Returns:
      sitk.Image: A label image with the same geometry as the input images.
    """
    if not seg_images:
        raise ValueError("No segmentation images provided.")

    # If labels are not provided, assign 1,2,3,...
    if labels is None:
        labels = list(range(1, len(seg_images) + 1))
    elif len(labels) != len(seg_images):
        raise ValueError("Length of labels must match number of segmentation images.")

    # Initialize the combined image with zeros.
    combined = sitk.Image(seg_images[0].GetSize(), sitk.sitkUInt8)
    combined.CopyInformation(seg_images[0])

    for seg, label in zip(seg_images, labels):
        # Ensure that the segmentation is binary.
        # Here we assume that any voxel > 0 belongs to the segmentation.
        binary_mask = sitk.BinaryThreshold(
            seg, lowerThreshold=1, upperThreshold=1e9, insideValue=1, outsideValue=0
        )
        # Multiply the binary mask by the label.
        label_img = sitk.Cast(binary_mask, sitk.sitkUInt8) * label

        # Combine using maximum. In case of overlaps, the highest label is taken.
        combined = sitk.Maximum(combined, label_img)

    return combined


def gaps(image, multilabel=False):
    """
    Show gaps in a binary or a multilabel segmentation
    """
    itk = _image()
    bin_im = binarise(image) if multilabel else image
    bin_full = itk.morph_operations(bin_im, "fill")

    # subtract the binarised image from the filled image
    return itk.image_operation("xor", bin_full, bin_im)


def fill_gaps(image1, image2=None, multilabel_images=False):
    """
    Fill gaps in a binary or a multilabel segmentation
        - Filling gaps in image1 ignoring gaps in image2
    """
    itk = _image()
    # if image2 is None :
    #     gaps_im = gaps(image1, multilabel=multilabel_images)
    # else :
    #     gaps_im = image_operation("xor", binarise(image1), binarise(image2))
    gaps_im = gaps(image1, multilabel=multilabel_images)
    if image2 is not None:
        gaps2 = gaps(image2, multilabel=multilabel_images)
        gaps_im = itk.image_operation("xor", gaps_im, gaps2)

    # get index where gaps is 1
    gaps_array_view = itk.imview(gaps_im)
    gaps_indices = np.argwhere(gaps_array_view == 1)

    number_of_gaps = gaps_indices.shape[0]
    if number_of_gaps == 0:
        logger.info("No gaps found.")
        return image1

    # convert to list of tuples for find_neighbours
    gaps_indices = list(map(tuple, gaps_indices))
    neighbours = itk.find_neighbours(image1, gaps_indices)

    logger.info(f"Found {number_of_gaps} gaps.")
    image1_array = itk.imarray(image1)
    for idx in gaps_indices:

        if len(neighbours[idx]) == 0:
            voxel_neighbours = itk.find_neighbours(image1, [idx])
            neighbours[idx] = voxel_neighbours[idx]

        # get all values associated with the neighbors of idx
        neighbour_values = [value[1] for value in neighbours[idx]]

        # get the most common value
        most_common_value = max(set(neighbour_values), key=neighbour_values.count)
        image1_array[idx[0], idx[1], idx[2]] = most_common_value

    filled_image = sitk.GetImageFromArray(image1_array)
    filled_image.CopyInformation(image1)

    return filled_image


def dice_score(true, pred):
    """Dice score between two label images.

    ``imview`` returns a *view* that does not own the image's buffer, so the
    ``sitk.Image`` must outlive it. The view is bound to a new name (rather than
    over the parameter) so the parameter keeps the image alive for the whole
    call — rebinding dropped the caller's last reference when the argument was a
    temporary, freeing the buffer while the view still pointed at it, and the sum
    then read whatever landed in that memory.
    """
    itk = _image()
    true_view = itk.imview(true)
    pred_view = itk.imview(pred)
    intersection = (true_view * pred_view).sum()
    return (2.0 * intersection) / (true_view.sum() + pred_view.sum())


def compare_images(im1: sitk.Image, im2: sitk.Image, return_comparison=False):
    """
    Returns a dictionary where the keys are the common labels between the two
    images and the values are the Dice scores for each label.
    It also returns a list of labels that are only present in one of the images.
    """
    labels_im1 = get_labels(im1)
    labels_im2 = get_labels(im2)

    common_labels = set(labels_im1).intersection(labels_im2)
    unique_labels = set(labels_im1).symmetric_difference(labels_im2)

    # find which image has the unique_labels
    unique_labels_im1 = unique_labels.intersection(labels_im1)
    unique_labels_im2 = unique_labels.intersection(labels_im2)

    unique_labels_dic = {"im1": unique_labels_im1, "im2": unique_labels_im2}

    scores = {}
    # sort common_labels to ensure consistent results
    common_labels = sorted(list(common_labels))

    for label in common_labels:
        im1_label = extract_single_label(im1, label, binarise=True)
        im2_label = extract_single_label(im2, label, binarise=True)
        scores[label] = dice_score(im1_label, im2_label)

    return scores, unique_labels_dic


def multilabel_comparison(
    im1: sitk.Image, im2: sitk.Image, l1: list = None, l2: list = None
) -> sitk.Image:
    """
    Compare two multilabel images and return a new image with the following values:
        - 1 if the label is present in both images
        - 2 if the label is present in im1 but not in im2
        - 3 if the label is present in im2 but not in im1
    """
    itk = _image()
    if l1 is None:
        l1 = get_labels(im1)
    if l2 is None:
        l2 = get_labels(im2)

    common_labels = set(l1).intersection(l2)
    # unique_labels = set(l1).symmetric_difference(l2)

    new_array = itk.imarray(im1)

    for label in common_labels:
        im1_label = extract_single_label(im1, label, binarise=True)
        im2_label = extract_single_label(im2, label, binarise=True)

        imc = sitk.And(im1_label, im2_label)
        imc_array = itk.imview(imc)
        new_array[imc_array > 0] = 0

    new_im = sitk.GetImageFromArray(new_array)
    new_im.CopyInformation(im1)

    return new_im


def split_label_into_components(image, label: int, open_image=False, open_radius=3):
    """
    Split one label into its connected components: the largest component keeps the
    original ``label``; the others get ``label*10 + i`` (escalating powers of 10 to
    avoid colliding with labels already present). Returns the input unchanged if the
    label forms a single component.

    Migrated verbatim from itktools.split_labels_on_repeats (renamed for clarity).
    """
    itk = _image()
    forbidden_labels = get_labels(image)
    forbidden_labels.remove(label)

    image_label = extract_single_label(image, label, binarise=True)

    if open_image:
        logger.info(f"Opening image with radius {open_radius}")
        image_label = itk.morph_operations(image_label, "open", radius=open_radius)

    cc_im_label, cc_labels, num_cc_labels = bwlabeln(image_label)
    if num_cc_labels == 1:
        logger.info(f"No connected components found for label {label}")
        return image

    logger.info(f"Found {num_cc_labels} connected components for label {label}")
    image_array = itk.imarray(image)
    image_array[np.equal(image_array, label)] = 0  # remove

    cc_array = itk.imview(cc_im_label)
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


def exchange_many_labels(input_image, old_labels: list, new_labels: list):
    labels_in_image = get_labels(input_image)

    swap_ops = get_labels_to_exchange(old_labels, new_labels, labels_in_image)
    new_image = _image().cp_image(input_image)
    for old_label, new_label in swap_ops:
        logger.info(f"Exchanging label {old_label} with {new_label}")
        new_image = exchange_labels(new_image, old_label, new_label)

    return new_image


def get_labels_volumes(im: sitk.Image) -> dict:
    """
    Quantifies the volumes of the labels of an image in units cubed.
    """

    # Ensure the image is of integer type
    segmentation_image = sitk.Cast(im, sitk.sitkUInt32)

    # Get the spacing of the image (voxel size)
    spacing = segmentation_image.GetSpacing()

    # Calculate the volume of a single voxel
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    # Use LabelStatisticsImageFilter to compute statistics for each label
    label_stats_filter = sitk.LabelStatisticsImageFilter()
    label_stats_filter.Execute(segmentation_image, segmentation_image)

    # Get the list of labels
    labels = label_stats_filter.GetLabels()

    # Initialize a dictionary to store the volume of each label
    label_volumes = {}

    # Iterate over each label and calculate its volume
    for label in labels:
        # Get the number of voxels for the current label
        voxel_count = label_stats_filter.GetCount(label)

        # Calculate the volume by multiplying voxel count with voxel volume
        label_volume = voxel_count * voxel_volume

        # Store the result in the dictionary
        label_volumes[label] = label_volume

    return label_volumes


def distance_based_outlier_detection(mlseg: sitk.Image, label=1, gauss_sigma=2.0) -> sitk.Image:
    """
    Find pointy bits of the segmentation based on distance to smooth version of itself
    """
    from imatools.io.image_io import save_image  # noqa: PLC0415

    itk = _image()
    segmentation = extract_single_label(mlseg, label, binarise=True)

    gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian_filter.SetSigma(gauss_sigma)  # Adjust sigma based on segmentation resolution
    smoothed_segmentation = gaussian_filter.Execute(segmentation)

    # cast smoothed_segmentation to the same pixel type as segmentation
    smoothed_segmentation = sitk.Cast(smoothed_segmentation, segmentation.GetPixelID())

    distance_map = sitk.Abs(segmentation - smoothed_segmentation)
    save_image(
        distance_map, "distance_map.nrrd", overwrite=True
    )  # Category-B bug: stray CWD write; preserved as-is
    sharp_regions = sitk.BinaryThreshold(distance_map, lowerThreshold=10, upperThreshold=1000)

    segmentation_array = itk.imarray(segmentation)
    sharp_array = itk.imview(sharp_regions)

    highlighted_segmentation = np.where(
        sharp_array == 1, 2, segmentation_array
    )  # Label sharp regions as '2'

    return itk.array2im(highlighted_segmentation, mlseg)


# ---------------------------------------------------------------------------
# Moved from imatools.common.itktools (M2a-2; zero-caller-but-KEEP function)
# ---------------------------------------------------------------------------


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
