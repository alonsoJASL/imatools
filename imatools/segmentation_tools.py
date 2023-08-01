import os
import argparse

from common import itktools as itku
from common import config  

logger = config.configure_logging(log_name=__name__) 

def main(args): 
    im_path = args.input_image
    # name remove extension
    im_name = os.path.basename(im_path)
    name, _ = os.path.splitext(im_name)
    if '.nii' in name:
        name, _ = os.path.splitext(name)

    base_dir = os.path.dirname(im_path)

    outname = name if args.output_name == "" else args.output_name
    outname += '.nii' if '.nii' not in outname else '' 
    input_image = itku.load_image(im_path)

    if args.mode == "extract":
        if args.label == -1:
            # find all the labels in image and extract them all into different files 
            labels = itku.get_labels(input_image)
            for label in labels:
                this_im = itku.extract_single_label(input_image, label, args.binarise)
                itku.save_image(this_im, base_dir, f'{outname}_label_{label}.nii')

        else:
            labels = args.label
            for label in labels:
                label_image = itku.extract_single_label(input_image, label, args.binarise)
                itku.save_image(label_image, base_dir, f'{outname}_label_{label}.nii')

    elif args.mode == "merge":
        merge_labels = args.merge_labels
        if merge_labels == -1:
            logger.error("Error: No labels to merge. Set them with the -m flag.")
            return 1

        merge_labels_str = list(map(str, merge_labels))
        label_images = []
        for label in merge_labels:
            label_images.append(itku.extract_single_label(input_image, label, args.binarise))

        merged_image = itku.merge_label_images(label_images)
        output_name = f'{outname}_merged_{"_".join(merge_labels_str)}.nii'
        itku.save_image(merged_image, base_dir, output_name)

    elif args.mode == "split":
        if -1 in args.label:
            labels = itku.get_labels(input_image)
        else :
            labels = args.label 

        for l in labels: 
            logger.info(f'Processing label: {l}')
            input_image = itku.split_labels_on_repeats(input_image, label=l, open_image=True, open_radius=args.split_radius)

        itku.save_image(input_image, base_dir, outname) 
    
    elif args.mode == "gaps":
        gaps = itku.find_gaps(input_image, multilabel_images=True)
        if args.output_name == "":
            outname = f'{name}_gaps.nii'

        itku.save_image(gaps, base_dir, outname)

    elif args.mode == "fill":

        old_segmentation = None if args.gaps_image == "" else itku.load_image(args.gaps_image)
        filled = itku.fill_gaps(input_image, old_segmentation, multilabel_images=True)
        
        if args.output_name == "":
            outname = f'{name}_filled.nii'

        itku.save_image(filled, base_dir, outname)

    elif args.mode == "morph":
        logger.info(f'Performing morphological operation: {args.morphological} on image: {im_path}')
        itku.save_image(itku.morph_operations(input_image, args.morphological, radius=args.split_radius, kernel_type='ball'), base_dir, outname) 

    elif args.mode == "show":
        itku.show_labels(input_image)

    elif args.mode == "inr":
        itku.convert_to_inr(input_image, os.path.join(base_dir, f'{outname}.inr'))


if __name__ == "__main__":
    input_parser = argparse.ArgumentParser(description="Extracts a single label from a label map image.")
    input_parser.add_argument("mode", choices=["extract", "merge", "split", "show", "gaps", "fill", "inr", "morph"], help="The mode to run the script in.")
    input_parser.add_argument("--morphological", "-morph", choices=["open", "close", "fillholes", "dilate", "erode", ""], default="", required=False, help="The operation to perform.")
    input_parser.add_argument("--input-image", "-in", required=True, help="The input image.")
    input_parser.add_argument("--gaps-image", "-gaps-in", default="", required=False, help="The input image to compare to gaps.")
    input_parser.add_argument("--label", "-l", type=int, nargs="+", default=-1, help="The label to extract. Default: -1 (all labels)")
    input_parser.add_argument("--output-name", "-out", default="", type=str, help="The output image prefix. (Default: <input_image>_label_<label_value>)")
    input_parser.add_argument("--merge-labels", "-m", nargs="+", type=int, default=-1, help="The labels to merge.")
    input_parser.add_argument("--binarise", "-bin", action="store_true", help="Binarise the label image.")
    input_parser.add_argument("--split-radius", "-radius", type=int, default=3, help="[MODE=split] Radius of morphological element")
    
    args = input_parser.parse_args()
    main(args)




