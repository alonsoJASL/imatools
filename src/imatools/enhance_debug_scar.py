import os
import numpy as np
import argparse
import SimpleITK as sitk

def fileparts(path):
    path = os.path.normpath(path)
    path_split = path.split(os.sep)
    out_dir = ''
    out_name = ''
    if len(path_split) == 1:
        path_split = path.split('/')
    if len(path_split) == 1:
        path_split = path.split('\\')

    if len(path_split) == 1:
        out_name = path_split[0]
    else :
        out_dir = os.sep.join(path_split[:-1])
        out_name = path_split[-1]

    return out_dir, out_name 

def get_threshold_values(thresholds, mean_bp, std_bp, method) :
    if method == 'iir' :
        threshold_values = [th * mean_bp for th in thresholds] 
    else: 
        threshold_values = [mean_bp + std_bp*th for th in thresholds]

    return threshold_values

def main(args):

    im_path = args.input
    _, im_name = fileparts(im_path)

    debug_scar_path = args.scar_corridor_image
    debug_scar_dir, _ = fileparts(debug_scar_path)

    prod_stats_path = args.image_info_file

    if '.nii' in im_name:
        im_name = os.path.splitext(im_name)[0]

    # Read image
    im = sitk.ReadImage(im_path)
    scar = sitk.ReadImage(debug_scar_path)

    if im.GetSize() != scar.GetSize():
        raise ValueError('Image and scar corridor image have different sizes')

    # Read image info from lines 1 and 2
    with open(prod_stats_path, 'r') as f:
        lines = f.readlines()
        mean_bp = float(lines[1])
        std_bp = float(lines[2])

    threshold_str = list(map(str, np.multiply(args.threshold, 100))) # remove decimal point 
    threshold_values = get_threshold_values(args.threshold, mean_bp, std_bp, args.threshold_method)
    output_name = f'enhanced_debug_{"_".join(threshold_str).replace(".","")}'

    im_array = sitk.GetArrayFromImage(im)
    scar_array = sitk.GetArrayFromImage(scar)
    enhanced_array = np.copy(scar_array)

    for x in range(scar_array.shape[0]):
        for y in range(scar_array.shape[1]):
            for z in range(scar_array.shape[2]): 
                scar_value = scar_array[x, y, z]
                lge_value = im_array[x, y, z]

                if scar_value > 1: 
                    enhanced_value=2
                    for th in threshold_values : 
                        enhanced_value += 1 if lge_value > th else 0 

                    enhanced_array[x, y, z] = enhanced_value 
            
    enhanced_scar = sitk.GetImageFromArray(enhanced_array)
    enhanced_scar.SetOrigin(scar.GetOrigin())
    enhanced_scar.SetSpacing(scar.GetSpacing())
    enhanced_scar.SetDirection(scar.GetDirection())

    enhanced_labels = np.unique(enhanced_array).tolist()
    enhanced_labels.remove(0) # Remove background
    enhanced_labels.remove(1) # Remove corridor 

    label_counter = [0] * len(enhanced_labels)
    total_counter = 0
    for x in range(enhanced_array.shape[0]):
        for y in range(enhanced_array.shape[1]):
            for z in range(enhanced_array.shape[2]): 
                value = enhanced_array[x, y, z]
                if value > 1: 
                    total_counter += 1
                if value in enhanced_labels:
                    label_counter[enhanced_labels.index(value)] += 1
    
    # write file with label counts 
    with open(os.path.join(debug_scar_dir, f'{output_name}_label_counts.txt'), 'w') as f:
        f.write(f'Total voxels: {total_counter}\n')
        for i, label in enumerate(enhanced_labels):
            f.write(f'{label}: {label_counter[i]}\n')

    # Save the new image
    sitk.WriteImage(enhanced_scar, os.path.join(debug_scar_dir, f'{output_name}.nii'))



if __name__ == '__main__':
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('--input', '-in', type=str, required=True, help='LGE image')
    input_parser.add_argument('--scar-corridor-image', '-scar-im', type=str, required=True, help='DEBUG scar image')
    input_parser.add_argument('--image-info-file', '-info', type=str, required=True, help='prodStats file')
    input_parser.add_argument('--threshold-method', '-m', choices=['iir', 'msd'], required=True, help='Threshold method')
    input_parser.add_argument('--threshold','-thres', nargs='+', type=float, required=True, help='Threshold value(s)')

    args = input_parser.parse_args()
    main(args)
