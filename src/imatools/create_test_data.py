import os
import argparse

from common import itktools as itku
from common import SegmentationGenerator as sg
from common import config  

logger = config.configure_logging(log_name=__name__)

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
    
    input_image = itku.load_image(im_path)
    return base_dir, name, input_image, outname, output_not_set

def execute_circle(args) : 
    if(args.help) : 
        print(execute_circle.__doc__)
        return

    outname = args.output_name if args.output_name != "" else "circle.nii"
    generator = sg.SegmentationGenerator(size=args.size, origin=args.origin, spacing=args.spacing)
    circle = generator.generate_circle(args.radius, args.center)

    itku.save_image(circle, outname)

def execute_cube(args) :
    if(args.help) : 
        print(execute_cube.__doc__)
        return

    outname = args.output_name if args.output_name != "" else "cube.nii"
    generator = sg.SegmentationGenerator(size=args.size, origin=args.origin, spacing=args.spacing)
    cube = generator.generate_cube(args.side, args.center)

    itku.save_image(cube, outname)

def main(args): 
    mode = args.mode
    if args.help == False and args.input_image == "":
        logger.error("Error: No input image. Set it with the -in flag.")
        return 1

    if mode == "circle":
        print("Circle mode")
    elif mode == "cube":
        print("Cube mode")
    elif mode == "lge":
        print("LGE mode")


if __name__ == "__main__":
    input_parser = argparse.ArgumentParser(description="Extracts a single label from a label map image.")
    input_parser.add_argument("mode", choices=["circle", "cube", "lge"], help="The mode to run the script in.")
    input_parser.add_argument("help", nargs='?', type=bool, default=False, help="Help page specific to each mode")
    input_parser.add_argument("-in", "--input-image", help="Input image path", type=str, default="")
    input_parser.add_argument("-out", "--output-name", help="Output image name", type=str, default="")
    input_parser.add_argument("-size", "--size", help="Size of the image", nargs=3, type=int, default=[300,300,100]) 
    input_parser.add_argument("--origin", nargs=3, type=float, help="Origin of the image", default=[0.0,0.0,0.0])
    input_parser.add_argument("--spacing", nargs=3, type=float, help="Spacing of the image", default=[1.0,1.0,1.0])

    lge_group = input_parser.add_argument_group("lge", "Arguments for lge mode")
    lge_group.add_argument("--lge-prism-size", nargs=3, type=int, help="Size of the prism", default=[80,80,80])
    lge_group.add_argument("--lge-method", type=str, choices=["iir", "msd"], help="Method of operation", default="iir")
    lge_group.add_argument("--lge-simple", action="store_true", help="Use simple scar generation method")

    group_circle = input_parser.add_argument_group("circle")
    group_circle.add_argument("-r", "--radius", help="Radius of the circle", type=int, default=80)
    group_circle.add_argument("-c", "--center", help="Center of the circle", nargs=3, type=int, default=[150,150,50])

    group_cube = input_parser.add_argument_group("cube")
    group_cube.add_argument("-s", "--side", help="Side of the cube", type=int, default=80)
    
    args = input_parser.parse_args()
    main(args)




