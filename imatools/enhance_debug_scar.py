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

    threshold_values = get_threshold_values(args.thresholds, mean_bp, std_bp, args.method) 

    enhanced_scar = sitk.Image(scar.GetSize(), sitk.sitkFloat32)
    enhanced_scar.SetOrigin(scar.GetOrigin())
    enhanced_scar.SetSpacing(scar.GetSpacing())
    enhanced_scar.SetDirection(scar.GetDirection())

    imiter = sitk.ImageRegionConstIterator(im)
    scariter = sitk.ImageRegionConstIterator(scar)
    enhancediter = sitk.ImageRegionConstIterator(enhanced_scar) 

    while not scariter.IsAtEnd() :
        scar_value = scariter.Get()
        lge_value = imiter.Get() 
        
        if scar_value > 1 :
            enhanced_value = 2
            for th in threshold_values : 
                enhanced_value += 1 if lge_value > th else 0 

            enhancediter.Set(enhanced_value)
        else :
            enhancediter.Set(scar_value) 

        imiter.Next()
        scariter.Next()
        enhancediter.Next() 
            
    # Save the new image
    sitk.WriteImage(enhanced_scar, os.path.join(debug_scar_dir, output_name))



if __name__ == '__main__':
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('--input', '-in', type=str, required=True, help='LGE image')
    input_parser.add_argument('--scar-corridor-image', '-scar-im', type=str, required=True, help='DEBUG scar image')
    input_parser.add_argument('--image-info-file', '-info', type=str, required=True, help='prodStats file')
    input_parser.add_argument('--threshold-method', '-m', choices=['iir', 'msd'], required=True, help='Threshold method')
    input_parser.add_argument('--threshold','-thres', nargs='+', type=float, required=True, help='Threshold value(s)')

    args = input_parser.parse_args()
    main(args)
