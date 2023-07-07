import os
import sys
import argparse

import SimpleITK as sitk
import nibabel as nib
import numpy as np

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

def save_image(image, dir, name):
    """
    Saves a SimpleITK image to disk.
    """
    sitk.WriteImage(image, os.path.join(dir, name))

def main(args) 
    im_path = args.input_image
    # name remove extension
    im_name = os.path.basename(im_path)
    name, _ = os.path.splitext(im_name)
    if '.nii' in name:
        name, _ = os.path.splitext(name)

    base_dir = os.path.dirname(im_path)

    outname = name if args.output_name == "" else args.output_name
    input_image = sitk.ReadImage(im_path)

    if args.mode == "extract":
        if args.label == -1:
            # find all the labels in image and extract them all into different files 
            labels = get_labels(input_image)
            for label in labels:
                this_im = extract_single_label(input_image, label, args.binarise)
                save_image(this_im, base_dir, f'{outname}_label_{label}.nii')

        else:
            labels = args.label
            for label in labels:
                label_image = extract_single_label(input_image, label, args.binarise)
                save_image(label_image, base_dir, f'{outname}_label_{label}.nii')

    elif args.mode == "merge":
        merge_labels = args.merge_labels
        if merge_labels == -1:
            print("Error: No labels to merge. Set them with the -m flag.")
            return 1

        merge_labels_str = list(map(str, merge_labels))
        label_images = []
        for label in merge_labels:
            label_images.append(extract_single_label(input_image, label, args.binarise))

        merged_image = merge_label_images(label_images)
        output_name = f'{outname}_merged_{"_".join(merge_labels_str)}.nii'
        save_image(merged_image, base_dir, output_name)

    elif args.mode == "show":
        show_labels(input_image)

    elif args.mode == "inr":
        convert_to_inr(input_image, os.path.join(base_dir, f'{outname}.inr'))


if __name__ == "__main__":
    input_parser = argparse.ArgumentParser(description="Extracts a single label from a label map image.")
    input_parser.add_argument("mode", choices=["extract", "merge", "show", "inr"], help="The mode to run the script in.")
    input_parser.add_argument("--input-image", "-in", required=True, help="The input image.")
    input_parser.add_argument("--label", "-l", type=int, nargs="+", default=-1, help="The label to extract. Default: -1 (all labels)")
    input_parser.add_argument("--output-name", "-out", default="", type=str, help="The output image prefix. (Default: <input_image>_label_<label_value>)")
    input_parser.add_argument("--merge-labels", "-m", nargs="+", type=int, default=-1, help="The labels to merge.")
    input_parser.add_argument("--binarise", "-bin", action="store_true", help="Binarise the label image.")
    
    args = input_parser.parse_args()
    main(args)




