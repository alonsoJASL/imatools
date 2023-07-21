import os
import sys
import argparse

import SimpleITK as sitk
import numpy as np

def load_image_as_np(path_to_file) :
    """ Reads image into numpy array """
    sitk_t1 = sitk.ReadImage(path_to_file)
    
    t1 = sitk.GetArrayFromImage(sitk_t1)
    origin = sitk_t1.GetOrigin()
    im_size = sitk_t1.GetSize()

    return t1, origin, im_size 

def load_image(path_to_file) :
    """ Reads image into SimpleITK object """
    sitk_t1 = sitk.ReadImage(path_to_file)
    return sitk_t1

def extract_single_label(image, label, binarise=False):
    """
    Extracts a single label from a label map image.
    """
    image_array = sitk.GetArrayFromImage(image)
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
    image_array = sitk.GetArrayFromImage(image)
    labels = np.unique(image_array).tolist()
    labels.remove(0) # background 
    l = [int(x) for x in labels]
    return l 


def save_image(image, dir_or_path, name=None):
    """
    Saves a SimpleITK image to disk.
    """
    output_path = dir_or_path if name is None else os.path.join(dir_or_path, name)
    sitk.WriteImage(image, output_path)