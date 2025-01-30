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
    
    itku.save_image(res_im, args.output)

def main(args) : 
    if args.mode == 'label_morph':
        label_morph(args)    
    else:
        logger.error('Unknown mode: {}'.format(args.mode))

if __name__ == "__main__":
    mychoices = ['label_morph']

    parser = argparse.ArgumentParser(description='Multilabel segmentation tools')
    parser.add_argument('mode', choices=mychoices, help='mode')
    parser.add_argument('-in', '--input', required=True, help='input image')
    parser.add_argument('-l', '--label', required=True, help='label image')
    parser.add_argument('-out', '--output', required=True, help='output image')
    
    group_morph = parser.add_argument_group('label_morph')
    group_morph.add_argument('--operation', choices=['dilate', 'erode', 'open', 'close', 'fill'], help='operation')
    group_morph.add_argument('--radius', type=int, help='radius', default=3)
    group_morph.add_argument('--kernel', choices=['ball', 'box', 'cross'], help='kernel', default='ball')

    args = parser.parse_args()
    main(args)

