import os
import argparse

from common import itktools as itku
from common import config  
from common import plotutils as pu

logger = config.configure_logging(log_name=__name__)

def main(args) : 
    mode = args.mode
    input_image_path = args.input
    label = args.label

    im = itku.load_image(input_image_path)
    label_mask = itku.extract_single_label(im, label, binarise=True)
    distinct_labels_image, labels_found, num_labels = itku.bwlabeln(label_mask)

    if mode == 'identify' :
        if label is None :
            logger.error('Label not provided')
            return
        
        logger.info(f'Label {label} found {num_labels} times.')

    elif mode == 'split' : 
        if label is None :
            logger.error('Label not provided')
            return
        
        logger.info('Splitting the label into distinct regions')
        if len(labels_found) > 1 : 
            new_labels = [(label*10 + ix) for ix in range(len(labels_found))]
            im_array = itku.imarray(im) 
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
        old_labels = args.old_labels
        new_labels = args.new_labels

        if len(old_labels) != len(new_labels) : 
            logger.error('Number of old labels and new labels should be the same')
            return
        
        new_image = itku.cp_image(im)
        for old_label, new_label in zip(old_labels, new_labels) : 
            new_image = itku.exchange_labels(new_image, old_label, new_label)
        
        output_path = os.path.join(os.path.dirname(input_image_path), args.output_name)
        itku.save_image(new_image, output_path)

        logger.info(f'Labels {old_labels} swapped with {new_labels}')
        logger.info(f'Output image saved at: {output_path}')


if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(description='Label Connectivity tools')
    parser.add_argument('mode', choices=['identify', 'split', 'swap'])
    parser.add_argument('-in', '--input', type=str, required=True)
    parser.add_argument('-label', '--label', type=int, default=None)
    parser.add_argument('-out', '--output-name', type=str, default=None)

    swap_labels_group = parser.add_argument_group('Swap Labels')
    swap_labels_group.add_argument('-old-labels', '--old-labels', type=int, nargs='+', default=None)
    swap_labels_group.add_argument('-new-labels', '--new-labels', type=int, nargs='+', default=None)

    args = parser.parse_args()
    main(args)