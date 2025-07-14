import os
import argparse

from common import itktools as itku
from common import config  
from common import plotutils as pu

logger = config.configure_logging(log_name=__name__)

def parse_input_path(input_path):
    """
    Parses the input path to extract the directory and file name.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The input path {input_path} does not exist.")
    
    im_dir = os.path.dirname(input_path)
    im_name = os.path.basename(input_path)
    
    return im_dir, im_name, input_path

def execute_canonical_orientation(args) : 
    im_dir, im_name, _ = parse_input_path(args.input)

    if args.output is None:
        args.output = os.path.join(im_dir, f"{im_name}_canonical.nrrd")

    itku.fix_header_and_save(input_path=args.input, output_path=args.output) 

def execute_reference_orientation(args):
    """
    Placeholder for restoring or like orientation functionality.
    """
    im_dir, im_name = parse_input_path(args.input)
    if args.secondary_input is None:
        raise ValueError("Secondary input is required for restore or like mode.")
    
    im = itku.load_image(args.input)
    ref = itku.load_image(args.secondary_input)

    if args.output is None:
        args.output = os.path.join(im_dir, f"{im_name}_restored.nrrd")

    new_im = itku.set_direction_as(im, ref)
    itku.save_image(new_im, args.output)

    
    
    
CHOICES = ['canonical', "restore", "like"]
def main(args):
    mode = args.mode
    if mode == "canonical":
        execute_canonical_orientation(args)
    elif mode == "restore" or mode == "like":
        
    else:
        logger.error(f"Unknown mode: {mode}. Please choose from {CHOICES}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical orientation script.")
    parser.add_argument("mode", choices=CHOICES, help="The mode to run the script in.")
    parser.add_argument("--input", "-in", type=str, help="Path to the input image file.")
    parser.add_argument("--output", "-out", type=str, default=None, help="Name of output (default: input_name_canonical.nrrd).")

    parser.add_argument("--secondary-input", "-in2", type=str, default=None, help="Path to a secondary input image file (optional).")
    
    args = parser.parse_args()
    main(args)
    