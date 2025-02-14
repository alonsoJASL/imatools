import os
import argparse

from common import itktools as itku
from common import config  
from common import plotutils as pu

logger = config.configure_logging(log_name=__name__)

MAX_INT_VALUE = 256

def ignore_labels_with_voxel_size_less_than(im, labels, min_voxel_size) : 
    logger.info(f'Ignoring labels with voxel size less than {min_voxel_size}')
    labels_to_ignore = []
    for ll in labels :
        mask = im == ll
        num_voxels = itku.get_num_nonzero_voxels(mask)
        if num_voxels < min_voxel_size : 
            labels_to_ignore.append(ll)
    
    return itku.exchange_many_labels(im, labels_to_ignore, [0]*len(labels_to_ignore))


def main(args) : 
    mode = args.mode
    input_image_path = args.input
    label = args.label

    im = itku.load_image(input_image_path)
    if label is not None :
        label_mask = itku.extract_single_label(im, label, binarise=True)
        distinct_labels_image, labels_found, num_labels = itku.bwlabeln(label_mask)
    elif mode in ['identify', 'split'] :
        logger.error('Label not provided')
        return
    

    if mode == 'identify' :
        if label is None :
            logger.error('Label not provided')
            return
        
        logger.info(f'Label {label} found {num_labels} times.')
        logger.info('Calculating volume of each region')
        spacing = itku.get_spacing(im)
        for ll in labels_found : 
            mask = distinct_labels_image == ll
            num_voxels = itku.get_num_nonzero_voxels(mask)
            logger.info(f'Volume of region {ll} is {num_voxels} voxels or {num_voxels*spacing[0]*spacing[1]*spacing[2]} mm^3')

    elif mode == 'split' : 
        if label is None :
            logger.error(f'MODE: {mode} requires a label, which was not provided. Use flag -l LABEL for it ')
            return
        
        logger.info('Splitting the label into distinct regions')
        if len(labels_found) > 1 : 
            im_array = itku.imarray(im) 

            if args.min_voxel_size is not None :
                distinct_labels_image = ignore_labels_with_voxel_size_less_than(distinct_labels_image, labels_found, args.min_voxel_size)
                labels_found = itku.get_labels(distinct_labels_image)
            
            labels_in_image = itku.get_labels(im)
            max_label_in_image = max(labels_in_image)
            new_labels = [max_label_in_image + (ix + 1) for ix in range(len(labels_found))]
            
            labels_array = itku.imview(distinct_labels_image)

            for ix, label in enumerate(labels_found) : 
                mask = labels_array == label
                new_label = new_labels[ix]
                im_array[mask] = new_label
            
            new_image = itku.array2im(im_array, im)
            output_path = os.path.join(os.path.dirname(input_image_path), args.output_name)
            itku.save_image(new_image, output_path)

            logger.info(f'Old label {label} split into {new_labels} distinct regions')
        else : 
            logger.info('No need to split the label into distinct regions')
            output_path = input_image_path

        logger.info(f'Output image saved at: {output_path}')    
    
    elif mode == 'swap' :
        logger.info('Swapping the labels')
        old_labels = args.old_labels if args.old_labels is not None else []
        new_labels = args.new_labels if args.new_labels is not None else []

        if args.old_label_range != '':
            old_labels_end_points = args.old_label_range.split(':')
            old_labels_range = list(range(int(old_labels_end_points[0]), int(old_labels_end_points[1])+1))
            new_labels_range_replace = [args.range_replace]*len(old_labels_range)

            old_labels.extend(old_labels_range)
            new_labels.extend(new_labels_range_replace)

        if len(old_labels) != len(new_labels) : 
            logger.error('Number of old labels and new labels should be the same')
            return
        
        new_image = itku.cp_image(im)
        for old_label, new_label in zip(old_labels, new_labels) : 
            logger.info(f'Swapping label {old_label} with {new_label}')
            new_image = itku.exchange_labels(new_image, old_label, new_label)
        
        output_path = os.path.join(os.path.dirname(input_image_path), args.output_name)
        itku.save_image(new_image, output_path)

        logger.info(f'Labels {old_labels} swapped with {new_labels}')
        logger.info(f'Output image saved at: {output_path}')

    elif mode == 'extract':  # extract sublabels from a label image
        logger.info('Extracting labels')

        folder = os.path.dirname(input_image_path)
        if args.min_voxel_size is not None :
            distinct_labels_image = ignore_labels_with_voxel_size_less_than(distinct_labels_image, labels_found, args.min_voxel_size)
            labels_found = itku.get_labels(distinct_labels_image)

        itku.save_image(distinct_labels_image, os.path.join(folder, f'labels_extracted_from_{label}.nrrd'))

    elif mode == 'regionprops' : 
        logger.info('Calculating region properties')
        region_properties = itku.regionprops(distinct_labels_image)
        logger.info(region_properties)


if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(description='Label Connectivity tools')
    parser.add_argument('mode', choices=['identify', 'split', 'extract', 'swap', 'regionprops'])
    parser.add_argument('-in', '--input', type=str, required=True)
    parser.add_argument('-label', '--label', type=int, default=None)
    parser.add_argument('-out', '--output-name', type=str, default=None)

    swap_labels_group = parser.add_argument_group('Swap Labels')
    swap_labels_group.add_argument('-old-labels', '--old-labels', type=int, nargs='+', default=None)
    swap_labels_group.add_argument('-new-labels', '--new-labels', type=int, nargs='+', default=None)
    swap_labels_group.add_argument('-old-label-range', '--old-label-range', type=str, default='')
    swap_labels_group.add_argument('-range-replace', '--range-replace', type=int, default=0)

    extract_split_labels_group = parser.add_argument_group('Extract/Split Labels')
    extract_split_labels_group.add_argument('-min-voxel-size', '--min-voxel-size', type=int, default=None)

    args = parser.parse_args()
    main(args)