import os
import argparse

from common import itktools as itku
from common import config  

logger = config.configure_logging(log_name=__name__)

def seg(x,y) : 
    res = f"seg_s{x}{y}.nrrd"

def rm_ext(name):
    return os.path.splitext(name)[0]

def get_base_inputs(args):
    im_path = args.input_image
    # name remove extension
    im_name = os.path.basename(im_path)
    name, _ = os.path.splitext(im_name)
    if '.nii' in name:
        name, _ = os.path.splitext(name)

    base_dir = os.path.dirname(im_path)

    output_not_set = (args.output_name == "")

    outname = name if output_not_set else args.output_name
    outname += '.nii' if '.nii' not in outname else '' 

    print(output_not_set, outname)

    input_image = itku.load_image(im_path)
    return base_dir, name, input_image, outname, output_not_set

def execute_compare(args):
    """
    Compares two label map images. Assumes they have the same labels.
    """
    if(args.help) : 
        print(execute_compare.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    if args.secondary_image == "":
        logger.error("Error: No image to compare. Set it with the -in2 flag.")
        return 1
    
    secondary_image = itku.load_image(args.secondary_image)
    scores, unique_labels = itku.compare_images(input_image, secondary_image) 

    for key in scores:
        print(f"{key}: {scores[key]}")
    
    print(f"Unique labels: {unique_labels}")


def main(args): 
    
        


if __name__ == "__main__":
    input_parser = argparse.ArgumentParser(description="Compare segmentation steps on two folders.")
    input_parser.add_argument("-dir", "--base-dir", help="Input segmentations path", type=str, default="")
    input_parser.add_argument("-gt", "--gt-dir", help="Input segmentations path", type=str, default="")
        
    
    args = input_parser.parse_args()
    main(args)




