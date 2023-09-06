import os
import sys
import json 
import argparse

from common import itktools as itku
from common import config

def chooseplatform():
    return sys.platform


# Constants and platform specific paths
# Change these accordingly!!
SCAR_CMD = {
    'linux': 'MitkCemrgScarProjectionOptions.sh',
    'win32': 'MitkCemrgScarProjectionOptions_release.bat',
}

CLIP_CMD = {
    'linux': 'MitkCemrgApplyExternalClippers',
    'win32': 'MitkCemrgApplyExternalClippers_release.bat',
}

MIRTK = {
    'linux': "$HOME/Desktop/CemrgApp-Linux-v2.2/CemrgApp-Linux/bin/MLib",
    'win32': "C:\Lib\cemrg_libraries\MLib"
}

logger = config.configure_logging(log_name=__name__)



def run_cmd(script_dir, cmd_name, arguments, debug=False):
    """ Return the command to execute"""
    cmd_name = os.path.join(script_dir, cmd_name) if script_dir != '' else cmd_name
    cmd = cmd_name + ' '
    cmd += ' '.join(arguments)
    stst = 0

    if debug:
        logger.info(cmd)
    else:
        stst = os.system(cmd)

    return stst, cmd


def create_segmentation_mesh(dir: str, pveins_file='LA-reg.nii', iterations=1, isovalue=0.5, blur=0.0, debug=False):
    arguments = [os.path.join(dir, pveins_file)]
    arguments.append(os.path.join(dir, 'segmentation.s.nii'))
    arguments.append('-iterations')
    arguments.append(str(iterations))
    seg_1_out, _ = run_cmd(MIRTK[chooseplatform()], 'close-image', arguments, debug)

    if seg_1_out != 0:
        logger.error('Error in close image')

    arguments.clear()
    arguments = [os.path.join(dir, 'segmentation.s.nii')]
    arguments.append(os.path.join(dir, 'segmentation.vtk'))
    arguments.append('-isovalue')
    arguments.append(str(isovalue))
    arguments.append('-blur')
    arguments.append(str(blur))
    seg_2_out, _ = run_cmd(MIRTK[chooseplatform()],
                           'extract-surface', arguments, debug)
    if seg_2_out != 0:
        logger.error('Error in extract surface')

    arguments.clear()
    arguments = [os.path.join(dir, 'segmentation.vtk')]
    arguments.append(os.path.join(dir, 'segmentation.vtk'))
    arguments.append('-iterations')
    arguments.append('10')
    seg_3_out, _ = run_cmd(MIRTK[chooseplatform()], 'smooth-surface', arguments, debug)
    if seg_3_out != 0:
        logger.error('Error in smooth surface')

    arguments.clear()


def scar_image_debug(image_size, prism_size, method, origin, spacing, simple):
    logger.info("Generating image of size {} with prism of size {} using method {}".format(image_size, prism_size, method))
    im, seg, boundic = itku.generate_scar_image(image_size, prism_size, origin, spacing, method, simple)
    return im, seg, boundic

def main(args):

    if args.mode == "testlge":
        image_size_tuple = tuple(args.lge_image_size)
        prism_size_tuple = tuple(args.lge_prism_size)
        origin_tuple = tuple(args.lge_origin)
        spacing_tuple = tuple(args.lge_spacing)
        im, seg, boundic = scar_image_debug(image_size_tuple, prism_size_tuple, args.lge_method, origin_tuple, spacing_tuple, args.lge_simple)

        if args.output is None:
            logger.info("No output file specified. Saving to default file name")
            output_path = os.path.dirname(__file__)
            output = "dcm-LGE_image_debug.nii"
        else:       
            output = args.output
            # get file path 
            output_path = os.path.dirname(output)
            output = os.path.basename(output)

        output_seg = "LA.nii"
        output_bounds = "bounds.json"

        # save image
        logger.info(f"Saving image to {output_path}")
        itku.save_image(im, output_path, output)
        itku.save_image(seg, output_path, output_seg)
        with open(os.path.join(output_path, output_bounds), "w") as f:
            json.dump(boundic, f)

    elif args.mode == "surf":
        if args.input is None:
            logger.error("No input file specified. Exiting...")
            return
        else:
            input_path = os.path.dirname(args.input)
            input_file = os.path.basename(args.input)

        # create segmentation mesh
        create_segmentation_mesh(input_path, input_file, debug=args.debug)


if __name__ == "__main__":
    input_parser = argparse.ArgumentParser(description="Segmentation tools for SCAR QUANTIFICATION")
    input_parser.add_argument("mode", type=str, choices=["testlge", "surf"], help="Mode of operation")
    input_parser.add_argument("-i", "--input", type=str, help="Input file name")
    input_parser.add_argument("-o", "--output", type=str, help="Output file name")
    input_parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    input_parser.add_argument("--lge-image-size", nargs=3, type=int, help="Size of the image", default=[300,300,100])
    input_parser.add_argument("--lge-prism-size", nargs=3, type=int, help="Size of the prism", default=[80,80,80])
    input_parser.add_argument("--lge-method", type=str, choices=["iir", "msd"], help="Method of operation", default="iir")
    input_parser.add_argument("--lge-origin", nargs=3, type=float, help="Origin of the image", default=[0.0,0.0,0.0])
    input_parser.add_argument("--lge-spacing", nargs=3, type=float, help="Spacing of the image", default=[1.0,1.0,1.0])
    input_parser.add_argument("--lge-simple", action="store_true", help="Use simple scar generation method")

    print("Running scarq_tools.py")

    args = input_parser.parse_args()
    main(args)

