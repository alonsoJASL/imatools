import os
import sys
import json 
import argparse

from common import itktools as itku
from common import vtktools as vtku
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

def get_io_folders(args) : 
    extract_base_dir = args.base_dir is None
    no_output_set = args.output is None
    if no_output_set :
        milog.info("No output file specified. Saving to default file name")
        output_path = os.path.dirname(__file__)
    else :
        output_path = os.path.dirname(args.output) if extract_base_dir else args.base_dir
    
    return extract_base_dir, no_output_set, output_path

def set_scarq_state(state_file:str, cemrg_dir: str, mirtk_dir: str, scar_cmd: str, clip_cmd: str) -> ScarQuantificationTools:
    """
    Set scarq state from a json file
    """

    if os.path.isfile(state_file) :
        scarq = ScarQuantificationTools()
        scarq.load_state(state_file)
        return scarq

    if cemrg_dir == "" : 
        cemrg_dir = CEMRG[chooseplatform()]
    if mirtk_dir == "" :
        mirtk_dir = MIRTK[chooseplatform()]
    scar_cmd += "_release.bat" if chooseplatform() == "win32" else ""
    clip_cmd += "_release.bat" if chooseplatform() == "win32" else ""
    
    scarq = ScarQuantificationTools(cemrg_folder=cemrg_dir, mirtk_folder=mirtk_dir, scar_cmd_name=scar_cmd, clip_cmd_name=clip_cmd)
    scarq.save_state(".","scarq_state.json")
    return scarq

def scar_image_debug(image_size, prism_size, method, origin, spacing, simple, help=False):
    """
    Create a test lge prism for debugging purposes. 
    """
    if help :
        print(scar_image_debug.__doc__)
        return None, None, None
    
    scarq = ScarQuantificationTools()
    im, seg, boundic = scarq.create_scar_test_image(image_size, prism_size, method, origin, spacing, simple)
    return im, seg, boundic

def execute_lge(args, help=False): 
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
        print(execute_lge.__doc__)
        return

    _, no_output_set, output_path = get_io_folders(args)

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

def execute_surf(args, help=False):
    """
    Creates segmentation.vtk surface mesh with MIRTK libraries

    MODE: surf

    Parameters:
    --base-dir (optional) : folder where all files are stored
    --input : Name of segmentation (default PVeinsCroppedImage.nii)
    --surf-iterations : Number of iterations (default 1)
    --surf-isovalue : Isovalue (default 0.5)
    --surf-blur : Blur (default 0.0)
    --clip-mitral-valve : Clip mitral valve name (default None)

    Usage:
    python scarq_tools.py surf --base-dir /path/to/data --input PVeinsCroppedImage.nii [--surf-iterations 1] [--surf-isovalue 0.5] [--surf-blur 0.0] [--clip-mitral-valve prodMVI.vtk]
    """
    if help :
        print(execute_surf.__doc__)
        return

    extract_base_dir, _, _ = get_io_folders(args)

    if args.input is None:
        dir = ""
        pveins_file = ""
    else :
        pveins_file = os.path.basename(args.input)
        dir = os.path.dirname(args.input) if extract_base_dir else args.base_dir

    if dir == "" or pveins_file == "":
        milog.error("No input file specified. Exiting...")
        return

    scarq = set_scarq_state(args.scarq_state, args.cemrg_dir, args.mirtk_dir, args.scar_cmd, args.clip_cmd)
    
    if scarq.check_mirtk(test=MIRTK_TEST) is False:
        milog.error(f"MIRTK not found in {scarq.mirtk}. Exiting...")
        return
    
    seg = itku.load_image(os.path.join(dir, pveins_file))
    if itku.check_for_existing_label(seg, 100) : 
        milog.info("Fixing segmentation's padding values before meshing")
        seg = itku.exchange_labels(seg, 100, 0)
        itku.save_image(seg, os.path.dirname(pveins_file), os.path.basename(pveins_file))

    iterations = args.surf_iterations
    isovalue = args.surf_isovalue
    blur = args.surf_blur
    debug = args.debug
    if args.surf_multilabel :
        milog.info("Using multilabel segmentation")
        labels = itku.get_labels(seg)
        for label in labels :
            milog.info(f"Creating mesh for label {label}")
            seg_label = itku.extract_single_label(seg, label)
            label_name = f"segmentation_{label}"

            itku.save_image(seg_label, dir, f"{label_name}.nii")
            scarq.create_segmentation_mesh(dir, f"{label_name}.nii", iterations, isovalue, blur, debug)
            os.rename(os.path.join(dir, "segmentation.vtk"), os.path.join(dir, f"{label_name}.vtk"))

            vtklabel = vtku.readVtk(os.path.join(dir, f"{label_name}.vtk"))
            vtklabel = vtku.set_cell_scalars(vtklabel, label)
            vtku.writeVtk(vtklabel, dir, f"{label_name}.vtk")

            if label == labels[0] :
                vtkout = vtku.readVtk(os.path.join(dir, f"{label_name}.vtk"))
            else :
                vtkout = vtku.join_vtk(vtkout, vtku.readVtk(os.path.join(dir, f"{label_name}.vtk")))
        
        vtku.writeVtk(vtkout, dir, "segmentation.vtk")
        
    else :
        scarq.create_segmentation_mesh(dir, pveins_file, iterations, isovalue, blur, debug)

        clip_mv = args.clip_mitral_valve
        if clip_mv is not None :
            scarq.clip_mitral_valve(dir, pveins_file, clip_mv)

    if args.surf_output != "segmentation" :
        milog.info(f"Renaming segmentation.vtk to {args.surf_output}.vtk")
        os.rename(os.path.join(dir, "segmentation.vtk"), os.path.join(dir, args.surf_output + ".vtk"))

def execute_scar_opts(args, help=False):
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
        print(execute_scar_opts.__doc__)
        return

    extract_base_dir, no_output_set, output_path = get_io_folders(args)
    input_path_file = extract_path(args.input, extract_base_dir, args.base_dir)
    dir = os.path.dirname(input_path_file)
    opts_file = os.path.basename(input_path_file) 
    output_dir = "OUTPUT" if no_output_set else os.path.basename(args.output)
    
    if dir == "" or opts_file == "":
        milog.error("No input file specified. Exiting...")
        return

    scarq = ScarQuantificationTools()
    scarq.create_scar_options_file(dir=dir, opt_file=opts_file, output_dir=output_dir, legacy=args.scar_opts_legacy)

def execute_scar(args, help=False): 
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
        print(execute_scar.__doc__)
        return

    extract_base_dir = get_io_folders(args)
    lge_path = extract_path(args.input, extract_base_dir, args.base_dir)
    if args.scar_opts is None : 
        args.scar_opts = ""

    seg_name = args.scar_seg
    opts_path = args.scar_opts
    svp = args.scar_opts_svp
    
    
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

    scarq = set_scarq_state(args.scarq_state, args.cemrg_dir, args.mirtk_dir, args.scar_cmd, args.clip_cmd)
    arguments = [
        "-i", lge_path,
        "-seg", seg_name,
        "-opts", opts_path
    ]

    scarq.run_scar(arguments)

def execute_mask(args, help=False):
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
        print(execute_mask.__doc__)
        return
    
    extract_base_dir, _, _ = get_io_folders(args)
    
    im2mask_path = extract_path(args.input, extract_base_dir, args.base_dir)
    mask_path = extract_path(args.mask, extract_base_dir, args.base_dir)
    thres_path = extract_path(args.mask_threshold_file, extract_base_dir, args.base_dir)
    output = extract_path(args.output, extract_base_dir, args.base_dir)

    im = itku.load_image(im2mask_path)
    mask = itku.load_image(mask_path)
    ignore_im_path=args.mask_ignore
    ignore_im = None if ignore_im_path == "" else itku.load_image(os.path.join(os.path.dirname(im2mask_path),ignore_im_path))

    thres_value=args.mask_threshold_value
    mask_value=args.mask_value
    seg2mask_path=args.mask_seg

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

def execute_point_or_cell(args, help=False) : 
    """
    Exchange point or cell data from a vtk file

    MODE: point2cell or cell2point

    Parameters:
    --base-dir (optional) : folder where all files are stored
    --input : input file name
    --output : output file name

    Usage:
    python scarq_tools.py point2cell --base-dir /path/to/data --input input.vtk --output output.vtk

    OR 

    python scarq_tools.py cell2point --base-dir /path/to/data --input input.vtk --output output.vtk
    """
    if help :
        print(execute_point_or_cell.__doc__)
        return
    
    extract_base_dir, _, _ = get_io_folders(args)

    vtk_path = extract_path(args.input, extract_base_dir, args.base_dir)
    output_path = extract_path(args.output, extract_base_dir, args.base_dir)
    
    function_dic = {
        "point2cell" : vtku.exchange_point_data_to_cell_data,
        "cell2point" : vtku.exchange_cell_data_to_point_data
    }
    vtkout = function_dic[args.mode](vtk_path)

    vtku.save_vtk(vtkout, output_path)

def main(args):

    myhelp = args.help
    if args.mode == "lge":
        execute_lge(args, help=myhelp)

    elif args.mode == "surf":
        execute_surf(args, help=myhelp)

    elif args.mode == "scar_opts" : 
        execute_scar_opts(args, help=myhelp)

    elif args.mode == "scar" :
        execute_scar(args, help=myhelp)

    elif args.mode == "mask" : 
        execute_mask(args, help=myhelp)

    elif args.mode == "point2cell" or args.mode == "cell2point" :
        execute_point_or_cell(args, help=myhelp)

    
if __name__ == "__main__":
    input_parser = argparse.ArgumentParser(description="Segmentation tools for SCAR QUANTIFICATION")
    input_parser.add_argument("mode", type=str, choices=["lge", "surf", "scar_opts", "scar" , "mask", "point2cell", "cell2point"], help="Mode of operation")
    input_parser.add_argument("help", nargs='?', type=bool, default=False, help="Help page specific to each mode")

    general_args = input_parser.add_argument_group("General arguments")
    general_args.add_argument("-dir", "--base-dir", type=str, help="Base directory")
    general_args.add_argument("-i", "--input", type=str, help="Input file name")
    general_args.add_argument("-o", "--output", type=str, help="Output file name")
    general_args.add_argument("-d", "--debug", action="store_true", help="Debug mode")

    lge_group = input_parser.add_argument_group("lge", "Arguments for lge mode")
    lge_group.add_argument("--lge-image-size", nargs=3, type=int, help="Size of the image", default=[300,300,100])
    lge_group.add_argument("--lge-prisexecute_scar_optsm-size", nargs=3, type=int, help="Size of the prism", default=[80,80,80])
    lge_group.add_argument("--lge-method", type=str, choices=["iir", "msd"], help="Method of operation", default="iir")
    lge_group.add_argument("--lge-origin", nargs=3, type=float, help="Origin of the image", default=[0.0,0.0,0.0])
    lge_group.add_argument("--lge-spacing", nargs=3, type=float, help="Spacing of the image", default=[1.0,1.0,1.0])
    lge_group.add_argument("--lge-simple", action="store_true", help="Use simple scar generation method")

    surf_group = input_parser.add_argument_group("surf", "Arguments for surf mode")
    surf_group.add_argument("--surf-multilabel", action="store_true", help="Use multilabel segmentation")
    surf_group.add_argument("--surf-iterations", type=int, help="Number of iterations", default=1)
    surf_group.add_argument("--surf-isovalue", type=float, help="Isovalue", default=0.5)
    surf_group.add_argument("--surf-blur", type=float, help="Blur", default=0.0)
    surf_group.add_argument("--clip-mitral-valve", type=str, default=None)
    surf_group.add_argument("--flip-xy", action="store_true", help="Flip xy axis")
    surf_group.add_argument("--surf-output", type=str, help="Output file name", default="segmentation")

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

    scarq_group = input_parser.add_argument_group("scarq", "Arguments for scarq mode")
    scarq_group.add_argument("--scarq-state", type=str, help="State file name", default="")
    scarq_group.add_argument("--cemrg-dir", type=str, help="CEMRG directory", default="")
    scarq_group.add_argument("--mirtk-dir", type=str, help="MIRTK directory", default="")
    scarq_group.add_argument("--scar-cmd", type=str, help="Scar command name", default="MitkCemrgScarProjectionOptions")
    scarq_group.add_argument("--clip-cmd", type=str, help="Clip command name", default="MitkCemrgApplyExternalClippers")

    milog.info("Running scarq_tools.py")

    args = input_parser.parse_args()
    main(args)

