import os
import argparse

from common import itktools as itku
from common import config  
from common import plotutils as pu

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
    is_nrrd = '.nrrd' in outname
    is_nii = '.nii' in outname
    if not is_nrrd and not is_nii:
        outname += '.nii' 
    
    input_image = itku.load_image(im_path)
    return base_dir, name, input_image, outname, output_not_set

def execute_extract(args):
    """
    Extracts a single label from a label map image.

    USAGE:
        python segmentation_tools.py extract -in <input_image> -l <label> [-out <output_name>]
    """
    if(args.help) : 
        print(execute_extract.__doc__)
        return
    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    
    if args.label == -1:
        # find all the labels in image and extract them all into different files 
        labels = itku.get_labels(input_image)
        for label in labels:
            this_im = itku.extract_single_label(input_image, label, args.binarise)
            outn = outname if output_not_set else f'{rm_ext(outname)}_label_{str(label)}.nii'
            itku.save_image(this_im, base_dir, outn)
    else:
        labels = args.label
        for label in labels:
            label_image = itku.extract_single_label(input_image, label, args.binarise)
            outn = outname if output_not_set else f'{rm_ext(outname)}_label_{str(label)}.nii'
            itku.save_image(label_image, base_dir, outn)
        
def execute_mask(args):
    """
    Masks a label map image with another label map image.

    USAGE:
        python segmentation_tools.py mask -in <input_image> -in2 <secondary_image> [-out <output_name>]
    """
    if(args.help) : 
        print(execute_mask.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    if args.secondary_image == "":
        logger.error("Error: No image to mask. Set it with the -in2 flag.")
        return 1
    secondary_image = itku.load_image(args.secondary_image)
    ignore_image = None if args.mask_ignore == "" else itku.load_image(args.mask_ignore)
    
    mask_name = os.path.basename(args.secondary_image)
    masked_image = itku.mask_image(input_image, secondary_image, args.mask_value, ignore_image)

    if output_not_set:
        outname = f'{rm_ext(outname)}_masked_{mask_name}.nii'

    itku.save_image(masked_image, base_dir, outname)

def execute_relabel(args):
    """
    Relabels a label map image.

    USAGE:
        python segmentation_tools.py relabel -in <input_image> -l <label> [-out <output_name>]
    """
    if(args.help) : 
        print(execute_relabel.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    if args.label == -1:
        logger.error("Error: No label to relabel. Set it with the -l flag.")
        return 1

    label = args.label
    relabelled_image = itku.relabel_image(input_image, label)
    
    if output_not_set:
        outname = f'{rm_ext(outname)}_relabelled_{label}.nii'
    print(output_not_set, outname)
    itku.save_image(relabelled_image, base_dir, outname)

def execute_remove(args):
    """
    Removes a label from a label map image.

    USAGE:
        python segmentation_tools.py remove -in <input_image> -l <label> [-out <output_name>]
    """
    if(args.help) : 
        print(execute_remove.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    if args.label == -1:
        logger.error("Error: No label to remove. Set it with the -l flag.")
        return 1

    label = args.label
    removed_image = input_image
    labels_removed_str = ""
    for l in label:
        removed_image = itku.exchange_labels(removed_image, l, 0)
        labels_removed_str += f"{l}_"
    
    if output_not_set:
        outname = f'{rm_ext(outname)}_removed_{labels_removed_str}.nii'
    itku.save_image(removed_image, base_dir, outname)

def execute_merge(args):
    """
    Merges labels from a label map image.

    USAGE:
        python segmentation_tools.py merge -in <input_image> -m <merge_labels> [-out <outname>]
    """
    if(args.help) : 
        print(execute_merge.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    if args.merge_labels == -1:
        logger.error("Error: No labels to merge. Set them with the -m flag.")
        return 1

    merge_labels_str = list(map(str, args.merge_labels))
    label_images = []
    for label in args.merge_labels:
        label_images.append(itku.extract_single_label(input_image, label, args.binarise))

    merged_image = itku.merge_label_images(label_images)
    if output_not_set:
        outname = f'{rm_ext(outname)}_merged_{"_".join(merge_labels_str)}.nii'
    itku.save_image(merged_image, base_dir, outname)

def execute_merge(args):
    """
    Merges labels from a label map image.

    USAGE:
        python segmentation_tools.py merge -in <input_image> -m <merge_labels> [-out <outname>]
    """
    if(args.help) : 
        print(execute_merge.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    if args.merge_labels == -1:
        logger.error("Error: No labels to merge. Set them with the -m flag.")
        return 1

    merge_labels_str = list(map(str, args.merge_labels))
    label_images = []
    for label in args.merge_labels:
        label_images.append(itku.extract_single_label(input_image, label, args.binarise))

    merged_image = itku.merge_label_images(label_images)
    if output_not_set:
        outname = f'{rm_ext(outname)}_merged_{"_".join(merge_labels_str)}.nii'
    itku.save_image(merged_image, base_dir, outname)

def execute_split(args):
    """
    Splits labels from a label map image.

    USAGE:
        python segmentation_tools.py split -in <input_image> -l <label> [-out <outname>]
    """
    if(args.help) : 
        print(execute_split.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    if args.label == -1:
        logger.error("Error: No label to split. Set it with the -l flag.")
        return 1

    label = args.label
    split_image = itku.split_labels_on_repeats(input_image, label, open_image=True, open_radius=args.split_radius)
    if output_not_set:
        outname = f'{rm_ext(outname)}_split_{label}.nii'
    itku.save_image(split_image, base_dir, outname)

def execute_gaps(args):
    """
    Finds gaps in a label map image.

    USAGE:
        python segmentation_tools.py gaps -in <input_image> [-out <outname>]
    """
    if(args.help) : 
        print(execute_gaps.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    gaps = itku.find_gaps(input_image, multilabel_images=True)
    if output_not_set:
        outname = f'{rm_ext(outname)}_gaps.nii'
    itku.save_image(gaps, base_dir, outname)

def execute_add(args):
    """
    Adds two label map images together.

    USAGE:
        python segmentation_tools.py add -in <input_image> -in2 <secondary_image> [-out <outname>]
    """
    if(args.help) : 
        print(execute_add.__doc__)
        return

    base_dir, name, input_image, outname, output_not_set = get_base_inputs(args)
    if args.secondary_image == "":
        logger.error("Error: No image to add. Set it with the --secondary-image flag.")
        return 1

    secondary_image = itku.load_image(args.secondary_image)
    add_name = os.path.basename(args.secondary_image)
    added = itku.add_images(input_image, secondary_image)

    if output_not_set:
        outname = f'{rm_ext(name)}_added_{rm_ext(add_name)}.nii'

    itku.save_image(added, base_dir, outname)

def execute_fill(args):
    """
    Fills gaps in a label map image.

    USAGE:
        python segmentation_tools.py fill -in <input_image> [-out <outname>]
    """
    if(args.help) : 
        print(execute_fill.__doc__)
        return

    base_dir, name, input_image, outname, output_not_set = get_base_inputs(args)
    old_segmentation = None if args.secondary_image == "" else itku.load_image(args.secondary_image)
    filled = itku.fill_gaps(input_image, old_segmentation, multilabel_images=True)
    
    if output_not_set:
        outname = f'{rm_ext(name)}_filled.nii'

    itku.save_image(filled, base_dir, outname)

def execute_morph(args):
    """
    Performs a morphological operation on a label map image.

    USAGE:
        python segmentation_tools.py morph -in <input_image> -morph <morphological_operation> [-out <outname>]
    """
    if(args.help) : 
        print(execute_morph.__doc__)
        return

    base_dir, name, input_image, outname, output_not_set = get_base_inputs(args)
    morphed = itku.morph_operations(input_image, args.morphological, radius=args.split_radius, kernel_type='ball')
    
    if output_not_set:
        outname = f'{rm_ext(name)}_morphed_{args.morphological}.nii'

    itku.save_image(morphed, base_dir, outname)

def execute_op(args):
    """
    Performs an operation on two label map images.

    USAGE:
        python segmentation_tools.py op -in <input_image> -op <operation> -in2 <secondary_image> [-out <outname>]
    """
    if(args.help) : 
        print(execute_op.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    if args.secondary_image == "":
        logger.error("Error: No image to perform op operation on. Set it with the -in2 flag.")
        return 1
    
    # input_image = itku.relabel_image(input_image, 1)
    secondary_image = itku.load_image(args.secondary_image)
    # secondary_image = itku.relabel_image(secondary_image, 1)

    op = itku.image_operation(args.op, input_image, secondary_image)
    if output_not_set:
        outname = f'{args.op}.nii'

    itku.save_image(op, base_dir, outname)

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
    if args.swap_axes:
        secondary_image = itku.swap_axes(secondary_image, [0, 2])
    scores, unique_labels = itku.compare_images(input_image, secondary_image) 

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)

    if not output_not_set:
        imc = itku.multilabel_comparison(input_image, secondary_image)
        itku.save_image(imc, base_dir, outname)

    for key in scores:
        print(f"{key}: {scores[key]}")
    
    print(f"Unique labels: {unique_labels}")

def execute_resample(args): 
    """
    Resamples a label map image. 

    Parameters: 
        --resample-spacing
        --resample-sigma
        --resample-smth-threshold
        --resample-close

    USAGE:
        python segmentation_tools.py resample -in <input_image> [-out <outname>]
    """
    if(args.help) : 
        print(execute_resample.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    sp = args.resample_spacing
    sig = args.resample_sigma
    smth_threshold = args.resample_smth_threshold
    im_close = args.resample_close
    resampled_image = itku.resample_smooth_label(input_image, spacing=sp, sigma=sig, threshold=smth_threshold, im_close=im_close)

    if output_not_set:
        outname = f'{rm_ext(outname)}_resampled.nii'

    itku.save_image(resampled_image, base_dir, outname)

def execute_smooth(args):
    """
    Smooths a label map image. 

    Parameters: 
        --resample-spacing
        --resample-sigma
        --resample-smth-threshold
        --resample-close

    USAGE:
        python segmentation_tools.py smooth -in <input_image> --resample-sigma <sigma> [-out <outname>]
    """
    if(args.help) : 
        print(execute_smooth.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    sigma = args.resample_sigma
    smoothed_image = itku.resample_smooth_label(input_image, spacing=args.resample_spacing, sigma=args.resample_sigma, threshold=args.resample_smth_threshold, im_close=args.resample_close)

    if output_not_set:
        outname = f'{rm_ext(outname)}_smoothed.nii'

    itku.save_image(smoothed_image, base_dir, outname)

def execute_largest(args):
    """
    Finds the largest label in a label map image. 

    USAGE:
        python segmentation_tools.py largest -in <input_image>
    """
    if(args.help) : 
        print(execute_largest.__doc__)
        return

    base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
    largest = itku.extract_largest(input_image)

    if output_not_set:
        outname = f'{rm_ext(outname)}_largest.nii'
    
    itku.save_image(largest, base_dir, outname)

def execute_inr2itk(args): 
    if (args.help) : 
        print("python segmentation_tools.py inr2itk -in <input_image> -out <output_name>")
        return
    
    input_path = args.input_image
    itk_image = itku.convert_from_inr(input_path)

    base_dir = os.path.dirname(input_path)
    outname = args.output_name if args.output_name.endswith('.nii') else f'{args.output_name}.nii'

    itku.save_image(itk_image, base_dir, outname)
    
def execute_plot(args):
    if(args.help) : 
        print("python segmentation_tools.py plot -in <input_image> [--out <output_name>]")
        return

    im = itku.load_image(args.input_image)
    outname = args.output_name
    name = "SEGMENTATION"

    pu.visualise_3d_segmentation(im, outname, name)


def main(args): 
    mode = args.mode
    if args.help == False and args.input_image == "":
        logger.error("Error: No input image. Set it with the -in flag.")
        return 1

    if mode == "extract":
        execute_extract(args)
    
    elif mode == "mask":
        execute_mask(args)

    elif mode == "relabel":
        execute_relabel(args)

    elif mode == "remove":
        execute_remove(args)

    elif mode == "merge":
        execute_merge(args)

    elif mode == "split":
        execute_split(args)
    
    elif mode == "gaps":
        execute_gaps(args)

    elif mode == "add":
        execute_add(args)

    elif mode == "fill":
        execute_fill(args)

    elif mode == "morph":
        execute_morph(args) 

    elif mode == "op":
        execute_op(args)

    elif mode == "show":
        if(args.help) : 
            print("python segmentation_tools.py show -in <input_image>")

        _, _, input_image, _, _ = get_base_inputs(args)
        itku.show_labels(input_image)

    elif mode == "inr":
        if(args.help) :
            print("python segmentation_tools.py inr -in <input_image> -out <output_name>")
            return 
        base_dir, _, input_image, outname, output_not_set = get_base_inputs(args)
        if outname.endswith('.nii') == True :
            outname = outname.replace('.nii', '')
        itku.convert_to_inr(input_image, os.path.join(base_dir, f'{outname}.inr'))
    
    elif mode == "compare": 
        execute_compare(args)

    elif mode == "resample":
        execute_resample(args)

    elif mode == "smooth" : 
        execute_smooth(args)

    elif mode == "largest":
        execute_largest(args)

    elif mode == "plot":
        execute_plot(args)
        


if __name__ == "__main__":
    mychoices = ['extract', 'relabel', 'remove', 'mask', 'merge', 'split', 'show', 'gaps', 'add', 'fill', 'inr', 'inr2itk', 'op', 'morph', 'compare', 'resample', 'smooth', 'largest', 'plot']
    #
    input_parser = argparse.ArgumentParser(description="Extracts a single label from a label map image.")
    input_parser.add_argument("mode", choices=mychoices, help="The mode to run the script in.")
    input_parser.add_argument("help", nargs='?', type=bool, default=False, help="Help page specific to each mode")
    input_parser.add_argument("--morphological", "-morph", choices=["open", "close", "fillholes", "dilate", "erode", ""], default="", required=False, help="The operation to perform.")
    input_parser.add_argument("--op", "-op", choices=[ "add", "subtract", "multiply", "divide", "and", "or", "xor", ""], default="", required=False, help="The operation to perform.")
    input_parser.add_argument("--input-image", "-in", required=False, help="The input image.", default="")
    input_parser.add_argument("--secondary-image", "-in2", default="", required=False, help="The secondary input image (use in: add, mask, fill).")
    input_parser.add_argument("--label", "-l", type=int, nargs="+", default=-1, help="The label to extract. Default: -1 (all labels)")
    input_parser.add_argument("--output-name", "-out", default="", type=str, help="The output image prefix. (Default: <input_image>_label_<label_value>)")
    input_parser.add_argument("--mask-ignore", default="", required=False, help="The Image to ignore in the mask image.")
    input_parser.add_argument("--mask-value", default=0, required=False, help="The value to ignore in the mask image.")
    input_parser.add_argument("--merge-labels", "-m", nargs="+", type=int, default=-1, help="The labels to merge.")
    input_parser.add_argument("--binarise", "-bin", action="store_true", help="Binarise the label image.")
    input_parser.add_argument("--split-radius", "-radius", type=int, default=3, help="[MODE=split] Radius of morphological element")
    input_parser.add_argument("--swap-axes", "-swap", action="store_true", help="Swap the axes [0, 2] of the image.")
    resample_group = input_parser.add_argument_group("Resample mode")
    resample_group.add_argument("--resample-spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="[MODE=resample] The new spacing to resample to.")
    resample_group.add_argument("--resample-sigma", type=float, default=3.0, help="[MODE=resample] The sigma for the gaussian kernel.")
    resample_group.add_argument("--resample-smth-threshold", type=float, default=0.5, help="[MODE=resample] The threshold for smoothing.")
    resample_group.add_argument("--resample-close", action="store_true", help="[MODE=resample] Close the image after resampling.")
    
    args = input_parser.parse_args()
    main(args)




