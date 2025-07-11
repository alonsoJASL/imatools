import os 
import argparse 

from imatools.common import itktools as itku
from imatools.common import config 
logger = config.configure_logging(log_name=__name__)

def label_morph(args) : 
    im = itku.load_image(args.input)
    label_im = itku.extract_single_label(im, args.label, binarise=True)
    morph_im = itku.morph_operations(label_im, args.operation, radius=args.radius, kernel_type=args.kernel)

    res_im = itku.simple_mask(im, morph_im, mask_value=args.label)
    
    output_path = os.path.join(os.path.dirname(args.input), args.output)
    itku.save_image(res_im, output_path)

def sharp_regions(args) :
    im = itku.load_image(args.input)
    sharp_region = itku.distance_based_outlier_detection(im, args.label, gauss_sigma=args.gauss_sigma) 

    out_path = os.path.join(os.path.dirname(args.input), f'sharp_regions_{args.label}.nrrd')
    itku.save_image(sharp_region, out_path)

def combine_segmentations(args) :
    folder = args.input
    img_list_names = [] 
    for file in os.listdir(folder):
        if file.endswith('.nrrd'):
            logger.info(f'Found image: {file}')
            img_list_names.append(file)
    
    img_list = [itku.load_image(os.path.join(folder, img_name)) for img_name in img_list_names]
    combined = itku.combine_segmentations(img_list)
    out_path = os.path.join(folder, args.output)
    itku.save_image(combined, out_path)
        
    
def simple_mask(args) :
    original_im = itku.load_image(args.input)
    mask_im = itku.load_image(args.mask)
    label = args.label
    res_im = itku.simple_mask(original_im, mask_im, mask_value=label)
    out_path = os.path.join(os.path.dirname(args.input), args.output)
    itku.save_image(res_im, out_path)