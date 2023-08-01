import os

import SimpleITK as sitk
import numpy as np
import json

from imatools.common.config import configure_logging
logger = configure_logging(log_name=__name__) 

def load_image_as_np(path_to_file) :
    """ Reads image into numpy array """
    sitk_t1 = sitk.ReadImage(path_to_file)
    
    t1 = sitk.GetArrayFromImage(sitk_t1)
    origin = sitk_t1.GetOrigin()
    im_size = sitk_t1.GetSize()

    return t1, origin, im_size 

def load_image(path_to_file) :
    """ Reads image into SimpleITK object """
    logger.info(f'Loading image from {path_to_file}')
    sitk_t1 = sitk.ReadImage(path_to_file)
    return sitk_t1

def extract_single_label(image, label, binarise=False):
    """
    Extracts a single label from a label map image.
    """
    image_array = sitk.GetArrayViewFromImage(image)
    label_array = np.zeros(image_array.shape, dtype=np.uint8)
    label_array[np.equal(image_array, label)] = 1 if binarise else label
    label_image = sitk.GetImageFromArray(label_array)
    label_image.CopyInformation(image)
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

def binarise(image, background=0):
    """
    Returns an image with only 0s and 1s.
    """
    image_array = sitk.GetArrayFromImage(image)
    bin_array = np.zeros(image_array.shape, dtype=np.uint8) 
    bin_array[np.greater(image_array, background)] = 1
    binim = sitk.GetImageFromArray(bin_array) 
    binim.CopyInformation(image) 

    return binim

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

def explore_labels_to_split(image):
    """
    Returns list of labels that can be split into multiple labels
    """
    labels = get_labels(image)
    labels_to_split = []
    for label in labels :
        _, _, num_cc_labels = bwlabeln(extract_single_label(image, label, binarise=True))
        if num_cc_labels > 1 :
            labels_to_split.append(label)

    return labels_to_split

def split_labels_on_repeats(image, label:int, open_image=False, open_radius=3):
    """
    Returns new image where label that can be split are split into two distinct 
    labels. The largest object gets the original label, while the others get 
            label*10 + ix, for ix in range(1, num_splits)
    If any label is already present in image, then that label is 100*label + ix
    """
    forbidden_labels = get_labels(image)
    forbidden_labels.remove(label)

    image_label = extract_single_label(image, label, binarise=True)

    if open_image : 
        logger.info(f'Opening image with radius {open_radius}')
        image_label = morph_operations(image_label, "open", radius=open_radius)

    cc_im_label, cc_labels, num_cc_labels = bwlabeln(image_label) 
    if num_cc_labels==1 : 
        logger.info(f'No connected components found for label {label}')
        return image 

    logger.info(f'Found {num_cc_labels} connected components for label {label}')
    image_array = sitk.GetArrayFromImage(image)
    image_array[np.equal(image_array, label)] = 0 # remove 

    cc_array = sitk.GetArrayViewFromImage(cc_im_label)
    for ix, ccl in enumerate(cc_labels) :
        new_label = label if ix == 0 else label*10 + (ccl-1)

        qx = 1
        while new_label in forbidden_labels : 
            new_label = label*np.power(10, qx) + (ccl-1)
            qx += 1

        image_array[np.equal(cc_array, ccl)] = new_label

    new_image = sitk.GetImageFromArray(image_array) 
    new_image.CopyInformation(image) 

    return new_image 

def morph_operations(image, operation:str, radius=3, kernel_type='ball') :
    """
    Performs a morphological operation on a binary image with a binary ball of a given radius 
    """
    switcher_operation = {
        "dilate": sitk.BinaryDilate,
        "erode": sitk.BinaryErode,
        "open": sitk.BinaryMorphologicalOpening,
        "close": sitk.BinaryMorphologicalClosing,
        "fill": sitk.BinaryFillhole
    }
    switcher_kernel = {
        "ball": sitk.sitkBall,
        "box": sitk.sitkBox,
        "cross": sitk.sitkCross
    }

    which_operation = switcher_operation.get(operation, lambda: "Invalid operation")
    which_kernel = switcher_kernel.get(kernel_type, lambda: "Invalid kernel type")

    if operation == 'fill':
        return which_operation(image)

    return which_operation(image, kernelRadius=(radius, radius, radius), kernelType = which_kernel)

def swap_labels(im, old_label: int, new_label=1):
    """
    Swaps all instances of old_label with new_label in a label image.
    """

    im_array = sitk.GetArrayFromImage(im)
    im_array[np.equal(im_array, old_label)] = new_label

    new_image = sitk.GetImageFromArray(im_array)
    
    new_image.SetOrigin(im.GetOrigin())
    new_image.SetSpacing(im.GetSpacing())
    new_image.SetDirection(im.GetDirection())

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
    print(f'Converting image to {out_path}')
    # Get the image data as a NumPy array
    data = sitk.GetArrayViewFromImage(image)
    data = data.astype(np.uint8)  # Convert to uint8

    # Extract relevant image information
    xdim, ydim, zdim = data.shape
    vx, vy, vz = image.GetSpacing()

    # Prepare header information
    header = r"#INRIMAGE-4#{"
    header += f"\nXDIM={xdim}\nYDIM={ydim}\nZDIM={zdim}\nVDIM=1\n"
    header += f"VX={vx:.4f}\nVY={vy:.4f}\nVZ={vz:.4f}\n"
    header += "TYPE=unsigned fixed\nPIXSIZE=8 bits\nCPU=decm\n"
    header += "\n" * (252 - len(header))  # Fill remaining space with newlines

    header += "##}\n"  # End of header

    # Write to binary file
    with open(out_path, "wb") as file:
        file.write(header.encode(encoding='utf-8'))  # Write header as bytes
        file.write(data.tobytes())  # Write data as bytes

def get_labels(image):
    """
    Returns a list of labels in an image.
    """
    image_array_view = sitk.GetArrayViewFromImage(image)
    labels = set(image_array_view.flatten())
    labels.discard(0) # background
    
    return sorted(map(int, labels))

def zeros_like(image):
    """
    Returns a new image with the same size and spacing as the input image, but filled with zeros.
    """
    return sitk.Image(image.GetSize(), sitk.sitkUInt8)

def save_image(image, dir_or_path, name=None):
    """
    Saves a SimpleITK image to disk.
    """
    output_path = dir_or_path if name is None else os.path.join(dir_or_path, name)
    sitk.WriteImage(image, output_path)

def points_to_image(image, points, label=1, girth=2, points_are_indices=False) : 
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
    image_np = sitk.GetArrayFromImage(zeros_like(image))
    print(f"Image shape: {image_np.shape}")
    
    # Loop through each point and find the closest voxel in the image
    for point in points:
        x, y, z = point
        # Convert world coordinates to image indices (pixel coordinates)
        if not points_are_indices :
            print(f"Point: {point}")
            index = image.TransformPhysicalPointToIndex((x, y, z))
            print(f"Index: {index}")
            index_rounded = np.round(index).astype(int)
            print(f"Rounded index: {index_rounded}")
        else :
            index_rounded = np.round(point).astype(int)

        # Set the label for the closest voxel
        for xi in range(-girth, girth):
            for yi in range(-girth, girth):
                for zi in range(-girth, girth):
                    image_np[index_rounded[2] + zi, index_rounded[1] + yi, index_rounded[0] + xi] = label
    
    # count number of voxels with label
    # print(f"Number of voxels with label {label}: {np.count_nonzero(image_np == label)}")

    # Convert the modified numpy array back to a SimpleITK image
    modified_image = sitk.GetImageFromArray(image_np)
    modified_image.SetOrigin(image.GetOrigin())
    modified_image.SetSpacing(image.GetSpacing())
    modified_image.SetDirection(image.GetDirection())

    return modified_image


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
        for key, value in points.items():
            points.append(value)

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

def image_operation(operation, image1, image2=None) : 
    switcher_operation = {
        "add": sitk.Add,
        "subtract": sitk.Subtract,
        "multiply": sitk.Multiply,
        "divide": sitk.Divide,
        "and": sitk.And,
        "or": sitk.Or,
        "xor": sitk.Xor,
        "not": sitk.Not
    }

    if image2 is None :
        return switcher_operation.get(operation, lambda: "Invalid operation")(image1)
    else :
        return switcher_operation.get(operation, lambda: "Invalid operation")(image1, image2)
    
def gaps(image, multilabel=False) : 
    """
    Show gaps in a binary or a multilabel segmentation
    """ 

    bin = binarise(image) if multilabel else image 
    bin_full = morph_operations(bin, "fill")

    # subtract the binarised image from the filled image
    return image_operation("xor", bin_full, bin)

def fill_gaps(image1, image2=None, multilabel_images=False) : 
    """
    Fill gaps in a binary or a multilabel segmentation
        - Filling gaps in image1 ignoring gaps in image2 
    """ 
    # if image2 is None : 
    #     gaps_im = gaps(image1, multilabel=multilabel_images)
    # else :
    #     gaps_im = image_operation("xor", binarise(image1), binarise(image2))
    gaps_im = gaps(image1, multilabel=multilabel_images)
    if image2 is not None :
        gaps2 = gaps(image2, multilabel=multilabel_images)
        gaps_im = image_operation("xor", gaps_im, gaps2)

    # get index where gaps is 1
    gaps_array_view = sitk.GetArrayViewFromImage(gaps_im)
    gaps_indices = np.argwhere(gaps_array_view==1)
    
    number_of_gaps = gaps_indices.shape[0]
    if number_of_gaps == 0:
        logger.info("No gaps found.")
        return image1
    
    # convert to list of tuples for find_neighbours
    gaps_indices = list(map(tuple, gaps_indices))
    neighbours = find_neighbours(image1, gaps_indices)
    
    logger.info(f"Found {number_of_gaps} gaps.")
    image1_array = sitk.GetArrayFromImage(image1)
    for idx in gaps_indices:

        if len(neighbours[idx]) == 0:
            voxel_neighbours = find_neighbours(image1, [idx])
            neighbours[idx] = voxel_neighbours[idx]
        
        # get all values associated with the neighbors of idx
        neighbour_values = [value[1] for value in neighbours[idx]]
        
        # get the most common value
        most_common_value = max(set(neighbour_values), key=neighbour_values.count)
        image1_array[idx[0], idx[1], idx[2]] = most_common_value

    filled_image = sitk.GetImageFromArray(image1_array)
    filled_image.CopyInformation(image1)

    return filled_image


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
    image_array_view = sitk.GetArrayViewFromImage(image)

    # Define the 26-connectivity offsets in 3D space
    offsets = [
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1),  (-1, 0, 0),  (-1, 0, 1),
        (-1, 1, -1),  (-1, 1, 0),  (-1, 1, 1),
        (0, -1, -1),  (0, -1, 0),  (0, -1, 1),
        (0, 0, -1),   (0, 0, 1),   (0, 1, -1),
        (0, 1, 0),    (0, 1, 1),
        (1, -1, -1),  (1, -1, 0),  (1, -1, 1),
        (1, 0, -1),   (1, 0, 0),   (1, 0, 1),
        (1, 1, -1),   (1, 1, 0),   (1, 1, 1)
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
