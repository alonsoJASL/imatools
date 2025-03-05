import os
import argparse
import shutil

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

def label_morphological_op(input_path, label, operation, radius, kernel, output_name) :
    im = itku.load_image(input_path)
    label_im = itku.extract_single_label(im, label, binarise=True)
    morph_im = itku.morph_operations(label_im, operation, radius=radius, kernel_type=kernel)

    res_im = itku.simple_mask(im, morph_im, mask_value=label)
    output_path = os.path.join(os.path.dirname(input_path), output_name)
    itku.save_image(res_im, output_path)


def main(args) :
    if args.mode == 'chain':
        labels_list = args.labels
        operations_list = args.operations
        if len(labels_list) != len(operations_list):
            logger.error('Number of labels and operations should be the same')
            return
        
        num_ops = len(operations_list)
        radii_list = args.radii if args.radii else args.radius
        if type(radii_list) == int:
            radii_list = [radii_list] * num_ops
        elif len(radii_list) != num_ops:
            logger.error('Number of labels and radii should be the same')
            return
        
        base_dir = os.path.dirname(args.input)
        aux_image_path = os.path.join(base_dir, 'aux_image.nrrd')
        shutil.copy(args.input, aux_image_path)

        for op, l, r in zip(operations_list, labels_list, radii_list):
            print(f'Applying {op} to label {l} with radius {r}')
            label_morphological_op(aux_image_path, l, op, r, args.kernel, 'aux_image.nrrd')
        
        shutil.move(aux_image_path, os.path.join(base_dir, args.output))



    else: 
        label_morphological_op(args.input, args.label, args.mode, args.radius, args.kernel, args.output)

if __name__ == "__main__":
    switcher_dict = itku.MORPH_SWITCHER
    mode_choices = list(switcher_dict.keys()) + ['chain']

    parser = argparse.ArgumentParser(description='Multilabel segmentation tools')
    parser.add_argument('mode', choices=mode_choices, help='operation')
    parser.add_argument('-in', '--input', required=True, help='input image')
    parser.add_argument('-l', '--label', required=True, help='label image')
    parser.add_argument('-out', '--output', help='output image', default='output.nrrd')
    parser.add_argument('--radius', type=int, help='radius', default=3)
    parser.add_argument('--kernel', choices=list(itku.KERNEL_SWITCHER.keys()), help='kernel', default='ball')
    
    group_morph = parser.add_argument_group('Chain option for morphological operations')
    group_morph.add_argument('--operations', nargs='+', type=str, help='operations to be chained', default=[])
    group_morph.add_argument('--labels', nargs='+', type=int, help='labels for the operations', default=[])
    group_morph.add_argument('--radii', nargs='+', type=int, help='radii for the operations (default: 3)', default=[])

    args = parser.parse_args()
    main(args)

