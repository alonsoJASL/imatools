import os
import argparse

from common import itktools as itku
from common import config  
from common import plotutils as pu

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

def main(args) : 
    if args.mode == 'label_morph':
        label_morph(args)    
    elif args.mode == 'sharp_regions':
        sharp_regions(args)
    elif args.mode == 'combine':
        combine_segmentations(args)
    elif args.mode == 'simple_mask':
        simple_mask(args)
    elif args.mode == 'show_voxels':
        im = itku.load_image(args.input)
        vox, world, bboxes = itku.get_indices_from_label(im, args.label, get_voxel_bbox=True)
        logger.info('Voxels at label {}: {}'.format(args.label, vox))
        logger.info('World coordinates: {}'.format(world))
        logger.info('Bounding boxes: {}'.format(bboxes))
    else:
        logger.error('Unknown mode: {}'.format(args.mode))

if __name__ == "__main__":
    mychoices = ['label_morph', 'sharp_regions', 'combine', 'simple_mask', 'show_voxels']

    parser = argparse.ArgumentParser(description='Multilabel segmentation tools')
    parser.add_argument('mode', choices=mychoices, help='mode')
    parser.add_argument('-in', '--input', required=True, help='input image')
    parser.add_argument('-l', '--label', required=True, help='label image')
    parser.add_argument('-out', '--output', help='output image', default='output.nrrd')
    
    group_morph = parser.add_argument_group('label_morph')
    group_morph.add_argument('--operation', choices=['dilate', 'erode', 'open', 'close', 'fill'], help='operation')
    group_morph.add_argument('--radius', type=int, help='radius', default=3)
    group_morph.add_argument('--kernel', choices=['ball', 'box', 'cross'], help='kernel', default='ball')

    sharp_group = parser.add_argument_group('sharp_regions')
    sharp_group.add_argument('--gauss-sigma', '-gauss-sigma', type=float, help='Gaussian sigma', default=2.0)

    mask_group = parser.add_argument_group('simple_mask')
    mask_group.add_argument('--mask', '-mask', help='mask image')

    args = parser.parse_args()
    main(args)

