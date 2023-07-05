import os
import sys
import numpy as np
import argparse
import SimpleITK as sitk


def fileparts(path):
    path = os.path.normpath(path)
    path_split = path.split(os.sep)
    if len(path_split) == 1:
        path_split = path.split('/')
    if len(path_split) == 1:
        path_split = path.split('\\')
    if len(path_split) == 1:
        raise ValueError('Path does not contain any folder')

    return os.sep.join(path_split[:-1]), path_split[-1]

def process_voxel(voxel_lge, voxel_scar, threshold_values, bloodpool, method='iir'): 
    mbp = bloodpool[0]
    sbp = bloodpool[1]
    result_voxel = 0;
    for th in threshold_values : 
        test_value = (th * mbp) if method == 'iir' else (mbp + th * sbp)
        if voxel_lge >= test_value and voxel_scar > 0:
            result_voxel += 1
        else:
            break

    return result_voxel

def main(args):

    im_path = args.input
    im_dir, im_name = fileparts(im_path)

    debug_scar_path = args.scar_corridor_image
    debug_scar_dir, debug_scar_name = fileparts(debug_scar_path)

    prod_stats_path = args.image_info_file
    prod_stats_dir, prod_stats_name = fileparts(prod_stats_path)

    if '.nii' in im_name:
        im_name = os.path.splitext(im_name)[0]

    output_name = 'debug_scar_enhanced.nii'

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

    threshold_values = args.threshold

    enhanced_scar = sitk.Image(scar.GetSize(), sitk.sitkFloat32)
    enhanced_scar.SetOrigin(scar.GetOrigin())
    enhanced_scar.SetSpacing(scar.GetSpacing())
    enhanced_scar.SetDirection(scar.GetDirection())

    parallel_iterator = sitk.pmap(process_voxel, im, scar, threshold_values, [mean_bp, std_bp], args.threshold_method) 

    enhanced_scar_iterator = sitk.ImageRegionIterator(new_image)
    for new_value in parallel_iterator:
        enhanced_scar_iterator.Set(new_value)
        enhanced_scar_iterator.Next()
    
    # Save the new image
    sitk.WriteImage(new_image, os.path.join(debug_scar_dir, output_name))



if __name__ == '__main__':
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('--input', '-in', type=str, required=True, help='LGE image')
    input_parser.add_argument('--scar-corridor-image', '-scar-im', type=str, required=True, help='DEBUG scar image')
    input_parser.add_argument('--image-info-file', '-info', type=str, required=True, help='prodStats file')
    input_parser.add_argument('--threshold-method', '-m', choices=['iir', 'msd'], required=True, help='Threshold method')
    input_parser.add_argument('--threshold','-thres', nargs='+', type=float, required=True, help='Threshold value(s)')

    args = input_parser.parse_args()
    main(args)
