import os

import SimpleITK as sitk
import vtk 
import numpy as np
import json

import imatools.common.vtktools as vtku 
from imatools.common.config import configure_logging
logger = configure_logging(log_name=__name__) 

def load_image_as_np(path_to_file) :
    """ Reads image into numpy array """
    sitk_t1 = sitk.ReadImage(path_to_file)
    
    t1 = imarray(sitk_t1)
    origin = sitk_t1.GetOrigin()
    im_size = sitk_t1.GetSize()

    return t1, origin, im_size 

def load_image(path_to_file) :
    """ Reads image into SimpleITK object """
    logger.info(f'Loading image from {path_to_file}')
    sitk_t1 = sitk.ReadImage(path_to_file)
    return sitk_t1

def extract_single_label(image, label, binarise=False) -> sitk.Image:
    """
    Extracts a single label from a label map image.
    """
    image_array = imview(image)
    label = np.array(label, dtype=image_array.dtype).item()

    label_array = np.zeros(image_array.shape, dtype=image_array.dtype)
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
    image_array = imarray(image)
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
    image_array = imarray(image)
    image_array[np.equal(image_array, label)] = 0 # remove 

    cc_array = imview(cc_im_label)
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

    logger.info(f'Performing {operation} operation with radius {radius} and kernel type {kernel_type}')

    which_operation = switcher_operation.get(operation, lambda: "Invalid operation")
    which_kernel = switcher_kernel.get(kernel_type, lambda: "Invalid kernel type")

    if operation == 'fill':
        return which_operation(image)

    return which_operation(image, kernelRadius=(radius, radius, radius), kernelType = which_kernel)

def swap_labels(im, old_label: int, new_label=1):
    """
    Swaps all instances of old_label with new_label in a label image.
    """

    im_array = imarray(im)
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
    data = imview(image)
    spacing = image.GetSpacing()
    # make sure elements less than 1 are 0 
    dtype = data.dtype
    bitlen = data.dtype.itemsize*8
    if dtype == bool or dtype == np.uint8:
        btype = 'unsigned fixed'
    elif dtype == np.uint16:
        btype = 'unsigned fixed'
    elif dtype == np.int16:
        btype = 'signed fixed'    
    elif dtype == np.float32:
        btype = 'float'
    elif dtype == np.float64:
        btype = 'float'
    else:
        raise ValueError('Volume format not supported')

    logger.info(f'Data type: {dtype}. TYPE:{btype} PIXSIZE:{bitlen}')
    xdim, ydim, zdim = data.shape
    header = f"#INRIMAGE-4#{{\nXDIM={xdim}\nYDIM={ydim}\nZDIM={zdim}\nVDIM=1\nVX={spacing[0]:.4f}\nVY={spacing[1]:.4f}\nVZ={spacing[2]:.4f}\n"
    header += "SCALE=2**0\n" if btype == 'unsigned fixed' or btype == 'signed fixed' else ""
    header += f"TYPE={btype}\nPIXSIZE={bitlen} bits\nCPU=decm"
    header += "\n" * (252 - len(header))  # Fill remaining space with newlines
    header += "##}\n"  # End of header

    # Write to binary file
    with open(out_path, "wb") as file:
        file.write(header.encode(encoding='utf-8'))  # Write header as bytes
        file.write(data.tobytes())  # Write data as bytes

def convert_from_inr(inr_path):
    """
    Converts an INR file to a SimpleITK image.
    """
    with open(inr_path, "rb") as file:
        header = ""
        while True:
            line = file.readline().decode('utf-8')
            header += line
            if line.strip() == "##}":
                break
        
        # Parse header
        header_dict = {}
        for line in header.split('\n'):
            if '=' in line:
                key, value = line.split('=')
                header_dict[key.strip()] = value.strip()
        
        xdim = int(header_dict['XDIM'])
        ydim = int(header_dict['YDIM'])
        zdim = int(header_dict['ZDIM'])
        spacing = [float(header_dict['VX']), float(header_dict['VY']), float(header_dict['VZ'])]
        pixsize = int(header_dict['PIXSIZE'].split()[0])
        dtype = header_dict['TYPE']
        
        if dtype == 'unsigned fixed':
            if pixsize == 8:
                np_dtype = np.uint8
            elif pixsize == 16:
                np_dtype = np.uint16
        elif dtype == 'signed fixed':
            if pixsize == 16:
                np_dtype = np.int16
        elif dtype == 'float':
            if pixsize == 32:
                np_dtype = np.float32
            elif pixsize == 64:
                np_dtype = np.float64
        else:
            raise ValueError('Volume format not supported')
        
        # Read image data
        data = np.frombuffer(file.read(), dtype=np_dtype)
        data = data.reshape((xdim, ydim, zdim), order='F')  # Fortran order to match INR format
        
        # Convert to SimpleITK image
        image = sitk.GetImageFromArray(data)
        image.SetSpacing(spacing)
        
        return image

def get_labels(image : sitk.Image ) -> list:
    """
    Returns a list of labels in an image.
    """
    image_array_view = imview(image)
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
    logger.info(f"Saving image to [{output_path}]")
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
    image_np = imarray(zeros_like(image))
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
        res_im = switcher_operation.get(operation, lambda: "Invalid operation")(image1)
    else :
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
    gaps_array_view = imview(gaps_im)
    gaps_indices = np.argwhere(gaps_array_view==1)
    
    number_of_gaps = gaps_indices.shape[0]
    if number_of_gaps == 0:
        logger.info("No gaps found.")
        return image1
    
    # convert to list of tuples for find_neighbours
    gaps_indices = list(map(tuple, gaps_indices))
    neighbours = find_neighbours(image1, gaps_indices)
    
    logger.info(f"Found {number_of_gaps} gaps.")
    image1_array = imarray(image1)
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
    image_array_view = imview(image)

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

def get_scarq_boundaries(mode :str) :#

    iir = mode.lower() == 'iir'
    
    lowthres = 0.9 if iir else 1.1
    fibrosis = 0.975 if iir else 2.0 
    scar = 1.21 if iir else 2.2 
    ablation = 1.33 if iir else 3.2
    ceiling = 1.5 if iir else 4.0

    bounds = [
        (lowthres, fibrosis),
        (fibrosis, scar),
        (scar, ablation),
        (ablation, ceiling)
    ]

    return bounds 

def generate_scar_image(image_size=(300, 300, 100), prism_size=(80, 80, 80), origin=(0, 0, 0), spacing=(1.0, 1.0, 1.0), mode = 'iir', simple=False, mean_bp_theory=100, std_bp_theory=10) : 
    """
    Creates an 'LGE image with scar' for testing purposes.
    """
    print(f"Generating {'simple' if simple else 'realistic'} image of size {image_size} with prism of size {prism_size} using method {mode}")

    # Create an image with user-defined dimensions, origin, and spacing
    size_adjusted = (image_size[2], image_size[1], image_size[0])
    image = sitk.Image(size_adjusted, sitk.sitkInt32)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    
    # Set values inside the prism
    start_indx = [(image_size[i] - prism_size[i]) // 2 for i in range(3)]
    end_indx = [start_indx[i] + prism_size[i] for i in range(3)]

    std_bp_theory = std_bp_theory if not simple else 0
    thres = lambda x, mbp, stdb : (x*mbp) if mode.lower() == 'iir' else (x*stdb + mbp)

    random_background = np.random.randint(0, 20, size=size_adjusted, dtype=np.int32)
    values_inside_prism = np.random.normal(mean_bp_theory, std_bp_theory, size=prism_size).astype(np.int32)
    mean_bp = np.mean(values_inside_prism)
    std_bp = np.std(values_inside_prism)

    # Create an array with all voxels set to 100 initially
    voxel_values = imarray(image)
    print(f'image size: {image_size}')
    print(f'size_adjusted: {size_adjusted}')
    print(f'size random_background: {random_background.shape}')
    print(f'voxel vlues size: {voxel_values.shape}')

    voxel_values[0:image_size[2], 0:image_size[1], 0:image_size[0]] = random_background
    voxel_values[start_indx[0]:end_indx[0], start_indx[1]:end_indx[1], start_indx[2]:end_indx[2]] = values_inside_prism

    # Create a prism mask with the specified dimensions
    prism_mask = imarray(zeros_like(image))
    prism_mask[start_indx[0]:end_indx[0], start_indx[1]:end_indx[1], start_indx[2]:end_indx[2]] = 1

    # Create the boundary region of the prism
    boundary_mask = imarray(zeros_like(image))
    boundary_mask[start_indx[0] - 1:end_indx[0] + 1, start_indx[1] - 1:end_indx[1] + 1, start_indx[2] - 1:end_indx[2] + 1] = 1
    boundary_mask *= (1 - prism_mask)  # Exclude the inside of the prism

    total_boundary_mask = np.sum(boundary_mask)

    d = get_scarq_boundaries(mode)
    percentages = [0.99, 0.01] if simple else [0.6, 0.2, 0.15, 0.05]
    boundic = { int(100*perc): np.round(perc*total_boundary_mask).astype(np.int32) for perc in percentages }
    totals = [np.round(perc*total_boundary_mask).astype(np.int32) for perc in percentages]
    
    boundic['mean_bp'] = mean_bp
    boundic['std_bp'] = std_bp

    indices = np.argwhere(boundary_mask == 1)
    np.random.shuffle(indices)
    tx = 0
    for idx in indices:
        try : 
            _ = f'Assigning value to voxel {idx} with total {totals[tx]}'
        except IndexError : 
            break

        if simple :
            val = 1.05 if tx == 0 else 1.21
            assign_value = thres(val, mean_bp, std_bp).astype(np.int32)
        else :
            low_bound = thres(d[tx][0], mean_bp, std_bp).astype(np.int32) 
            high_bound = thres(d[tx][1], mean_bp, std_bp).astype(np.int32) 
            assign_value = np.random.randint(low_bound, high_bound, dtype=np.int32)
        
        voxel_values[idx[0], idx[1], idx[2]] = assign_value
        totals[tx] -= 1

        if totals[tx] == 0 :
            tx += 1
    
    # Set pixel values in the SimpleITK image
    out_image = sitk.GetImageFromArray(voxel_values)
    out_image.CopyInformation(image)

    boundic['total'] = np.sum(list(boundic.values()), dtype=np.int32)
    for key in boundic:
        if isinstance(boundic[key], np.int32):
            boundic[key] = int(boundic[key])
    
    logger.info(f"Boundaries: {boundic}")

    # Create a segmentation image
    segmentation = sitk.GetImageFromArray(prism_mask)
    segmentation.CopyInformation(image)
        
    return out_image, segmentation, boundic

def relabel_image(input_image, new_label) :
    """Assumes input_image is a binary image, every value>0 is set to new_label"""
    input_array = imarray(input_image)
    input_array[np.greater(input_array, 0)] = new_label

    new_image = sitk.GetImageFromArray(input_array)
    new_image.CopyInformation(input_image)

    return new_image

def cp_image(input_image) :
    """Copy an image"""
    input_array = imarray(input_image)

    new_image = sitk.GetImageFromArray(input_array)
    new_image.CopyInformation(input_image)

    return new_image

def exchange_labels(input_image, old_label, new_label) :
    input_array = imarray(input_image)
    input_array[np.equal(input_array, old_label)] = new_label

    new_image = sitk.GetImageFromArray(input_array)
    new_image.CopyInformation(input_image)

    return new_image

def exchange_many_labels(input_image, old_labels:list, new_labels:list) :
    input_array = imarray(input_image)
    labels_to_reprocess = []
    for old_l, new_l in zip(old_labels, new_labels) :
        if old_l == new_l :
            print(f'Old label {old_l} is the same as new label {new_l}. Skip...')
            continue

        # if new label exists in old labels, then set it to a new one, larger thatn all the old labels
        if new_l in old_labels :
            new_l_aux = max(old_labels) + 1
            print(f'New label {new_l} already exists in old labels. Setting it to a new label {new_l_aux}...')
            input_array[np.equal(input_array, new_l)] = new_l_aux
            input_array[np.equal(input_array, old_l)] = new_l

            labels_to_reprocess.append((new_l_aux, new_l))
        else :
            print(f'Exchanging label {old_l} with {new_l}...')
            input_array[np.equal(input_array, old_l)] = new_l

    for old_l, new_l in labels_to_reprocess :
        print(f'Exchanging label {old_l} with {new_l}...')
        input_array[np.equal(input_array, old_l)] = new_l

    new_image = sitk.GetImageFromArray(input_array)
    new_image.CopyInformation(input_image)

    return new_image

def exchange_labels_form_json(input_image, json_old: str, json_new: str) :
    with open(json_old, 'r') as f:
        old_labels_json = json.load(f)
    with open(json_new, 'r') as f:
        new_labels_json = json.load(f)

    old_labels = list(old_labels_json.values())
    new_labels = list(new_labels_json.values())

    return exchange_many_labels(input_image, old_labels, new_labels)
        
def add_images(im1, im2) :
    add_im = sitk.Add(im1, im2)
    add_im.CopyInformation(im1)

    return add_im

def simple_mask(im, mask, mask_value=0) -> sitk.Image :
    masked_im_array = imarray(im)
    mask_array = imarray(mask)
    
    masked_im_array[ mask_array > 0 ] = mask_value
    
    new_im = sitk.GetImageFromArray(masked_im_array)
    new_im.CopyInformation(im)

    return new_im

def get_mask_array_with_restrictions(im, mask, threshold=0, ignore_im=None) -> np.ndarray :
    mask_array = imarray(mask)
    if threshold > 0 :
        im_array = imview(im)
        mask_array[ im_array > threshold ] = 1
        
    if ignore_im is not None :
        ignore_im_array = imview(ignore_im)
        mask_array[ ignore_im_array > 0 ] = 0

    mask_array[ mask_array > 0 ] = 1
    return mask_array

def mask_image(im, mask, mask_value=0, ignore_im=None, threshold=0):
    masked_im_array = imarray(im)
    mask_array = get_mask_array_with_restrictions(im, mask, threshold=threshold, ignore_im=ignore_im)
    
    masked_im_array[ mask_array > 0 ] = mask_value
    
    new_im = sitk.GetImageFromArray(masked_im_array)
    new_im.CopyInformation(im)

    return new_im

def get_mask_with_restrictions(im, mask, threshold=0, ignore_im=None) : 
    """
    Returns a mask with the following restrictions: 
        - mask_value is set to 1
        - mask_value is set to 0 if ignore_im is not None and ignore_im > 0
        - mask_value is set to 0 if mask_value > threshold
    """
    mask_array = get_mask_array_with_restrictions(im, mask, threshold=threshold, ignore_im=ignore_im)
    new_mask = sitk.GetImageFromArray(mask_array)
    new_mask.CopyInformation(im)

    return new_mask

def check_for_existing_label(im: sitk.Image, label) -> bool :
    """
    Check if a particular label exists in an image
    """
    labels_in_im = get_labels(im)
    return (label in labels_in_im )

def create_normal_vector_for_plane(axis, angle) : 
    """
    Returns a normal vector for a plane rotated around the given axis by the given angle
    """
    AXES = ['x', 'y', 'z']
    if axis not in AXES : 
        raise ValueError(f'Axis {axis} not recognised')
    
    vector = np.zeros(3)
    vector[AXES.index(axis)] = 1

    angle_rad = np.radians(angle)

    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    # Apply the rotation matrix to the initial vector
    normal_vector = np.dot(rotation_matrix, vector)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector


def create_image_at_plane(image: sitk.Image, point_on_plane: np.array, axis:str, angle:float) :
    normal_vector = create_normal_vector_for_plane(axis, angle)
    return create_image_at_plane_from_vector(image, point_on_plane, normal_vector)

def create_image_at_plane_from_vector(image: sitk.Image, point_on_plane: np.array, normal_vector: np.array): 
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

def dice_score(true, pred):
    true = imview(true)
    pred = imview(pred)
    intersection = (true * pred).sum()
    return (2. * intersection) / (true.sum() + pred.sum())

def compare_images(im1: sitk.Image, im2: sitk.Image, return_comparison=False) :
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

    unique_labels_dic = {
        'im1': unique_labels_im1,
        'im2': unique_labels_im2
    }

    scores = {}
    # sort common_labels to ensure consistent results
    common_labels = sorted(list(common_labels))
    for label in common_labels:
        im1_label = extract_single_label(im1, label, binarise=True)
        im2_label = extract_single_label(im2, label, binarise=True)
        scores[label] = dice_score(im1_label, im2_label)
    
    return scores, unique_labels_dic

def multilabel_comparison(im1: sitk.Image, im2: sitk.Image, l1: list = None, l2: list = None) -> sitk.Image :
    """
    Compare two multilabel images and return a new image with the following values:
        - 1 if the label is present in both images
        - 2 if the label is present in im1 but not in im2
        - 3 if the label is present in im2 but not in im1
    """
    if l1 is None :
        l1 = get_labels(im1)
    if l2 is None :
        l2 = get_labels(im2)

    common_labels = set(l1).intersection(l2)
    # unique_labels = set(l1).symmetric_difference(l2)

    new_array = imarray(im1)

    for label in common_labels:
        im1_label = extract_single_label(im1, label, binarise=True)
        im2_label = extract_single_label(im2, label, binarise=True)

        imc = sitk.And(im1_label, im2_label)
        imc_array = imview(imc)
        new_array[imc_array > 0] = 0
        

    new_im = sitk.GetImageFromArray(new_array)
    new_im.CopyInformation(im1)

    return new_im

# def multilabel_curvature(im: sitk.Image, gradient_sigma=1.0) -> sitk.Image :
#     """
#     Calculate the curvature of a multilabel image.
#     """
#     unique_labels = get_labels(im)
#     im_size = im.GetSize()

#     pixel_type = sitk.sitkUInt8
#     output_im = sitk.Image(im_size, pixel_type)
#     output_im.CopyInformation(im)

#     for label in unique_labels:
#         # Create a binary image for the current label
#         binary_im = sitk.BinaryThreshold(im, lowerThreshold=label, upperThreshold=label)
#         binary_im.CopyInformation(im)

#         # Calculate the curvature of the binary image
#         curvature_im = sitk.CurvatureFlow(binary_im, timeStep=0.125, numberOfIterations=5)
#         curvature_im = sitk.BinaryThreshold(curvature_im, lowerThreshold=0.5, upperThreshold=1.5)

#         # Check for overlapping voxels and remove them from curvature_im
#         overlapping_voxels = sitk.And(output_im, sitk.Cast(curvature_im, pixel_type))
#         curvature_im = sitk.Subtract(sitk.Cast(curvature_im, pixel_type), overlapping_voxels)
#         curvature_im = sitk.Multiply(curvature_im, label)

#         # Add the resampled label image to the final result
#         output_im = sitk.Add(output_im, curvature_im)

#     return output_im

def resample_smooth_label(im: sitk.Image, spacing: list, sigma=3.0, threshold=0.5, im_close=True):
    # import itk

    # Get all unique labels in the image
    unique_labels = get_labels(im)
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

        resampled_label_im = sitk.BinaryThreshold(resampled_label_im, lowerThreshold=threshold, upperThreshold=2) 

        if im_close :
            resampled_label_im = morph_operations(resampled_label_im, "close")

        # Check for overlapping voxels and remove them from resampled_label_im
        overlapping_voxels = sitk.And(resampled_im, sitk.Cast(resampled_label_im, pixel_type))
        resampled_label_im = sitk.Subtract(sitk.Cast(resampled_label_im, pixel_type), overlapping_voxels)
        resampled_label_im = sitk.Multiply(resampled_label_im, label)

        # Add the resampled label image to the final result
        resampled_im = sitk.Add(resampled_im, resampled_label_im)

    return resampled_im

def smooth_labels(im: sitk.Image, sigma=1.0, threshold=0.5, im_close=True):
    unique_labels = get_labels(im)
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

def project_surface_onto_segmentation(segmentation: sitk.Image, surface: vtk.vtkPolyData, check_visited=False) -> vtk.vtkPolyData :
    cog = vtku.get_cog_per_element(surface)    
    scalars = surface.GetCellData().GetScalars()
    visited_indices = set()
    for ix in range(surface.GetNumberOfCells()) :
        x, y, z = cog[ix]
        value = scalars.GetTuple1(ix) 
        index = segmentation.TransformPhysicalPointToIndex((x,y,z))

        if visited_indices.__contains__(index) and check_visited :
            continue

        visited_indices.add(index)
        segmentation.SetPixel(index, value)

    return segmentation

def swap_axes(im: sitk.Image, axes: list) -> sitk.Image :
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

def segmentation_curvature(im: sitk.Image, gradient_sigma = 1.0) -> sitk.Image :
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

def segmentation_curvature_value(im: sitk.Image, gradient_sigma = 1.0) -> float :
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

def imarray(im: sitk.Image) -> np.ndarray :
    return sitk.GetArrayFromImage(im)

def imview(im: sitk.Image) -> np.ndarray :
    return sitk.GetArrayViewFromImage(im)

def extract_largest(im: sitk.Image) -> sitk.Image :
    """
    Extract the largest connected component from a multilabel image.
    """
    image_labels = get_labels(im)
    im_binary = binarise(im)
    cc = sitk.ConnectedComponent(im_binary)
    labels = get_labels(cc)

    im_pixel_type = im.GetPixelID()
    
    if len(labels) == 1:
        return im
    
    # Get the size of each connected component
    sizes = []
    for label in labels:
        sizes.append(np.sum(imarray(cc) == label))

    # Find the label with the largest size
    largest_label = labels[np.argmax(sizes)]
    largest_cc = sitk.BinaryThreshold(cc, lowerThreshold=largest_label, upperThreshold=largest_label)

    # cast the largest connected component to the same pixel type as the input image
    largest_cc = sitk.Cast(largest_cc, im_pixel_type)

    return sitk.Multiply(im, largest_cc)

def array2im(im_array: np.ndarray, im: sitk.Image) -> sitk.Image :
    """
    Convert a NumPy array to a SimpleITK image.
    """
    im_out = sitk.GetImageFromArray(im_array)
    im_out.CopyInformation(im)

    return im_out
