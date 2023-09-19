import os
import sys
import json 
import argparse

from common import itktools as itku
from common import config
from common.scarqtools import ScarQuantificationTools

def chooseplatform():
    return sys.platform

# Constants and platform specific paths
# Change these accordingly!!
SCAR_CMD = {
    'linux': 'MitkCemrgScarProjectionOptions',
    'win32': 'MitkCemrgScarProjectionOptions_release.bat',
}

CLIP_CMD = {
    'linux': 'MitkCemrgApplyExternalClippers',
    'win32': 'MitkCemrgApplyExternalClippers_release.bat',
}

MIRTK = {
    'linux': "/home/jsl19/syncdir/cemrgapp_prebuilds/v2018.04.2/linux/Externals/MLib",
    'win32': "C:\Lib\cemrg_libraries\MLib"
}

CEMRG = {
    'linux': "/home/jsl19/dev/build/CEMRG2-U20.04/MITK-build/bin",
    'win32': "C:/dev/build/CEMRG/MITK-build/bin"
}

MIRTK_TEST = 'close-image'
MIRTK_TEST += '.exe' if sys.platform == 'win32' else ''

milog = config.configure_logging(log_name=__name__)

def extract_path(some_str, extract_base_dir=False, base_dir="") -> str:
    if some_str is None :
        path = ""
    else:
        path = some_str if extract_base_dir else os.path.join(base_dir, some_str)
    
    return path

def create_segmentation_mesh(dir: str, pveins_file='PVeinsCroppedImage.nii', iterations=1, isovalue=0.5, blur=0.0, debug=False, help=False):
    """
    Creates segmentation.vtk surface mesh with MIRTK libraries

    MODE: surf

    Parameters:
    --base-dir (optional) : folder where all files are stored
    --input : Name of segmentation (default PVeinsCroppedImage.nii)
    --surf-iterations : Number of iterations (default 1)
    --surf-isovalue : Isovalue (default 0.5)
    --surf-blur : Blur (default 0.0)

    Usage:
    python scarq_tools.py surf --base-dir /path/to/data --input PVeinsCroppedImage.nii [--surf-iterations 1] [--surf-isovalue 0.5] [--surf-blur 0.0]
    """
    if help :
        print(create_segmentation_mesh.__doc__)
        return
    
    if dir == "" or pveins_file == "":
        milog.error("No input file specified. Exiting...")
        return

    scarq = ScarQuantificationTools(mirtk_folder=MIRTK[chooseplatform()])

    if scarq.check_mirtk(test=MIRTK_TEST) is False:
        milog.error(f"MIRTK not found in {scarq.mirtk}. Exiting...")
        return
    
    seg = itku.load_image(os.path.join(dir, pveins_file))
    if itku.check_for_existing_label(seg, 100) : 
        milog.info("Fixing segmentation's padding values before meshing")
        seg = itku.exchange_labels(seg, 100, 0)
        itku.save_image(seg, os.path.dirname(pveins_file), os.path.basename(pveins_file))

    scarq.create_segmentation_mesh(dir, pveins_file, iterations, isovalue, blur, debug)
  

def create_scar_options_file(dir: str, opts_file='options.json', output_dir = "OUTPUT", old = False, help=False) :
    """
    Creates a basic file with scar options

    MODE: scar_opts

    Parameters:
    --base-dir (optional) : folder where all files are stored
    --input : Name of options file (default options.json)
    --output : Name saved in options file (default OUTPUT)
    --scar-opts-legacy : Use legacy scar options (default False)

    Usage:
    python scarq_tools.py scar_opts --base-dir /path/to/data --input options.json [--output OUTPUT] [--scar-opts-legacy]
    """
    if help :
        print(create_scar_options_file.__doc__)
        return
    
    if dir == "" or opts_file == "":
        milog.error("No input file specified. Exiting...")
        return

    scarq = ScarQuantificationTools()
    scarq.create_scar_options_file(dir=dir, opt_file=opts_file, output_dir=output_dir, legacy=old)

def scar3d(lge_path: str, seg_name: str, opts_path: str, svp=False, help=False) :
    """
    Perform scar quantification on a 3D image
    .\MitkCemrgScarProjectionOptions_release.bat -i path_lge -seg name_seg -opts path_opts

    MODE: scar

    Parameters:
    --base-dir (optional) : folder where all files are stored
    --input : Path to LGE image 
    --scar-seg : Name of segmentation (default PVeinsCroppedImage.nii)
    --scar-opts : Path to options file (use mode scar_opts to create one)

    Usage:
    python scarq_tools.py scar --base-dir /path/to/data --input dcm-LGE-test.nii --scar-seg PVeinsCroppedImage.nii --scar-opts path/to/options.json
    """
    if help :
        print(scar3d.__doc__)
        return
    
    if lge_path == "" or opts_path == "":
        milog.error("No input files specified. Exiting...")
        return
    
    with open(opts_path, "r") as f:
        json_opts = json.load(f)
    
    if ("single_voxel_projection" not in json_opts.keys()) or (json_opts["single_voxel_projection"] != svp) : 
        milog.info(f"Rewritting [{opts_path}] to include single voxel projection option")
        json_opts["single_voxel_projection"] = svp
        with open(opts_path, "w") as f:
            json.dump(json_opts, f)

    scarq = ScarQuantificationTools(cemrg_folder=CEMRG[chooseplatform()], scar_cmd_name=SCAR_CMD[chooseplatform()], clip_cmd_name=CLIP_CMD[chooseplatform()])
    arguments = [
        "-i", lge_path,
        "-seg", seg_name,
        "-opts", opts_path
    ]

    scarq.run_scar(arguments)

def scar_image_debug(image_size, prism_size, method, origin, spacing, simple, help=False):
    """
    Create a test lge prism for debugging purposes. 

    MODE: lge

    Parameters: 
    --base-dir (optional) : folder where all files are stored 
    --output (optional) : output file name
    --lge-image-size : size of the image (default 300,300,100)
    --lge-prism-size : size of the prism (default 80,80,80)
    --lge-method : method of operation (iir or msd)
    --lge-origin : origin of the image (default 0.0,0.0,0.0) 
    --lge-spacing : spacing of the image (default 1.0,1.0,1.0)
    --lge-simple : creates simple image 

    Usage: 
    python scarq_tools.py lge --base-dir /path/to/data --output dcm-LGE-test.nii [--lge-method iir] [--lge-image-size 300 300 100] [--lge-prism-size 80 80 80] [--lge-origin 0.0 0.0 0.0] [--lge-spacing 1.0 1.0 1.0] [--lge-simple]
    
    """
    if help :
        print(scar_image_debug.__doc__)
        return None, None, None
    
    scarq = ScarQuantificationTools()
    im, seg, boundic = scarq.create_scar_test_image(image_size, prism_size, method, origin, spacing, simple)
    return im, seg, boundic

def mask_image(im2mask_path, mask_path, thres_path, thres_value=0, mask_value=0, ignore_im_path="", seg2mask_path="", output="", help=False):
    """
    Mask an image with another image, where voxels are above a certain threshold

    MODE: mask

    Parameters:
    --base-dir (optional) : folder where all files are stored
    --input : image to be masked (or considered for threshold (when using --mask-seg)))) 
    --mask-seg : segmentation to mask (default None)
    --mask : mask image
    --mask-threshold-file : threshold file (prodStats.txt file)
    --mask-threshold-value : threshold value (default 0.0)
    --mask-value : value to mask (default 0.0)
    --mask-ignore : ignore image (default None)
    --output : output file name (default masked.nii)

    Usage:
    python scarq_tools.py mask --base-dir /path/to/data --input dcm-LGE-test.nii --mask DebugScar.nii --mask-threshold-file prodStats.txt [--mask-threshold-value VALUE] [--mask-value VALUE] [--output masked.nii]
    """
    if help :
        print(mask_image.__doc__)
        return
    
    im = itku.load_image(im2mask_path)
    mask = itku.load_image(mask_path)
    ignore_im = None if ignore_im_path == "" else itku.load_image(os.path.join(os.path.dirname(im2mask_path),ignore_im_path))

    scarq = ScarQuantificationTools()
    meanbp, stdbp = scarq.get_bloodpool_stats_from_file(thres_path)
    if seg2mask_path == "" :
        masked_im = scarq.mask_voxels_above_threshold(im, mask, meanbp, stdbp, thres_value, mask_value, ignore_im)
    else :
        seg2mask_path = os.path.join(os.path.dirname(im2mask_path), seg2mask_path)
        masked_im = scarq.mask_segmentation_above_threshold(seg2mask_path, im, mask, meanbp, stdbp, thres_value, mask_value, ignore_im)

    if output == "" :
        opath = os.path.dirname(im2mask_path)
        ofile = "masked.nii"
        output = os.path.join(opath, ofile)
    
    itku.save_image(masked_im, output)

    return None

def main(args):

    extract_base_dir = args.base_dir is None
    no_output_set = args.output is None
    myhelp = args.help
    if no_output_set :
        milog.info("No output file specified. Saving to default file name")
        output_path = os.path.dirname(__file__)
    else :
        output_path = os.path.dirname(args.output) if extract_base_dir else args.base_dir

    if args.mode == "lge":
        image_size_tuple = tuple(args.lge_image_size)
        prism_size_tuple = tuple(args.lge_prism_size)
        origin_tuple = tuple(args.lge_origin)
        spacing_tuple = tuple(args.lge_spacing)
        im, seg, boundic = scar_image_debug(image_size_tuple, prism_size_tuple, args.lge_method, origin_tuple, spacing_tuple, args.lge_simple, help=myhelp)

        if im is not None : 
            output = "dcm-LGE_image_debug.nii" if no_output_set else os.path.basename(args.output)
            output_seg = "LA.nii"
            output_bounds = "bounds.json"

            # save image
            milog.info(f"Saving image to {output_path}")
            itku.save_image(im, output_path, output)
            itku.save_image(seg, output_path, output_seg)
            with open(os.path.join(output_path, output_bounds), "w") as f:
                json.dump(boundic, f)

    elif args.mode == "surf":

        if args.input is None:
            input_path = ""
            input_file = ""
        else :
            input_file = os.path.basename(args.input)
            input_path = os.path.dirname(args.input) if extract_base_dir else args.base_dir

        # create segmentation mesh
        create_segmentation_mesh(input_path, input_file, debug=args.debug, help=myhelp)

    elif args.mode == "scar_opts" : 
        input_path_file = extract_path(args.input, extract_base_dir, args.base_dir)
        input_path = os.path.dirname(input_path_file)
        input_file = os.path.basename(input_path_file) 

        output = "OUTPUT" if no_output_set else os.path.basename(args.output)

        create_scar_options_file(input_path, opts_file=input_file, output_dir = output, old = args.scar_opts_legacy, help=myhelp)

    elif args.mode == "scar" :
        lge_path = extract_path(args.input, extract_base_dir, args.base_dir)

        if args.scar_opts is None : 
            args.scar_opts = ""
        
        scar3d(lge_path, args.scar_seg, args.scar_opts, args.scar_opts_svp, help=myhelp)

    elif args.mode == "mask" : 
        im2mask_path = extract_path(args.input, extract_base_dir, args.base_dir)
        mask_path = extract_path(args.mask, extract_base_dir, args.base_dir)
        thres_path = extract_path(args.mask_threshold_file, extract_base_dir, args.base_dir)
        output_path = extract_path(args.output, extract_base_dir, args.base_dir)

        mask_image(im2mask_path, mask_path, thres_path, args.mask_threshold_value, args.mask_value, args.mask_ignore, args.mask_seg, output_path, help=myhelp)
        

    
if __name__ == "__main__":
    input_parser = argparse.ArgumentParser(description="Segmentation tools for SCAR QUANTIFICATION")
    input_parser.add_argument("mode", type=str, choices=["lge", "surf", "scar_opts", "scar" , "mask"], help="Mode of operation")
    input_parser.add_argument("help", nargs='?', type=bool, default=False, help="Help page specific to each mode")

    general_args = input_parser.add_argument_group("General arguments")
    general_args.add_argument("-dir", "--base-dir", type=str, help="Base directory")
    general_args.add_argument("-i", "--input", type=str, help="Input file name")
    general_args.add_argument("-o", "--output", type=str, help="Output file name")
    general_args.add_argument("-d", "--debug", action="store_true", help="Debug mode")

    lge_group = input_parser.add_argument_group("lge", "Arguments for lge mode")
    lge_group.add_argument("--lge-image-size", nargs=3, type=int, help="Size of the image", default=[300,300,100])
    lge_group.add_argument("--lge-prism-size", nargs=3, type=int, help="Size of the prism", default=[80,80,80])
    lge_group.add_argument("--lge-method", type=str, choices=["iir", "msd"], help="Method of operation", default="iir")
    lge_group.add_argument("--lge-origin", nargs=3, type=float, help="Origin of the image", default=[0.0,0.0,0.0])
    lge_group.add_argument("--lge-spacing", nargs=3, type=float, help="Spacing of the image", default=[1.0,1.0,1.0])
    lge_group.add_argument("--lge-simple", action="store_true", help="Use simple scar generation method")

    surf_group = input_parser.add_argument_group("surf", "Arguments for surf mode")
    surf_group.add_argument("--surf-iterations", type=int, help="Number of iterations", default=1)
    surf_group.add_argument("--surf-isovalue", type=float, help="Isovalue", default=0.5)
    surf_group.add_argument("--surf-blur", type=float, help="Blur", default=0.0)

    scar_group = input_parser.add_argument_group("scar", "Arguments for scar mode")
    scar_group.add_argument("--scar-seg", type=str, help="Segmentation file name", default="PVeinsCroppedImage.nii")
    scar_group.add_argument("--scar-opts", type=str, help="Options path")
    scar_group.add_argument("--scar-opts-svp", action="store_true", help="Single voxel projection option")
    scar_group.add_argument_group("scar_opts", "Arguments for scar_opts mode")

    scar_opts_group = input_parser.add_argument_group("scar_opts", "Arguments for scar_opts mode")
    scar_opts_group.add_argument("--scar-opts-legacy", action="store_true", help="Use legacy scar options")

    mask_group = input_parser.add_argument_group("mask", "Arguments for mask mode")
    mask_group.add_argument("--mask-seg", type=str, help="Segmentation file name", default="")
    mask_group.add_argument("--mask", type=str, help="Mask file name", default="")
    mask_group.add_argument("--mask-threshold-file", type=str, help="Threshold file name", default="")
    mask_group.add_argument("--mask-threshold-value", type=float, help="Threshold value", default=0.0)
    mask_group.add_argument("--mask-value", type=float, help="Mask value", default=0.0)
    mask_group.add_argument("--mask-ignore", type=str, help="Ignore image path file", default="")

    milog.info("Running scarq_tools.py")

    args = input_parser.parse_args()
    main(args)

