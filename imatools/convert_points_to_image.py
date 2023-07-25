import os
import argparse

from common import itktools as itku
from common import ioutils as iou

def main(args) : 
    input_path = args.input
    points_path = args.points
    label = args.label
    output_name = args.output
    output_name += ".nii" if not output_name.endswith(".nii") else ""
    output_path = os.path.join(os.path.dirname(input_path), output_name)

    modified_image = itku.pointfile_to_image(input_path, points_path, label, points_are_indices=args.indices)
    itku.save_image(modified_image, output_path)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Convert coordinates to index in the image.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument("-pts", "--points", type=str, required=True, help="Path to the coordinates file.")
    parser.add_argument("-l", "--label", type=int, default=1, help="Label value to set for the closest voxels to the points.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output image.")
    parser.add_argument("-indx", "--indices", action="store_true", help="If set, the points are indices in the image, not world coordinates.")
    parser.add_argument("-g", "--girth", type=int, default=2, help="The girth of the cube around the point to set to the label value.")
    
    args = parser.parse_args()
    main(args)
