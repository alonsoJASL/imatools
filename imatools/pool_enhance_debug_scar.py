import os
import numpy as np
import argparse
import SimpleITK as sitk
import multiprocessing

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

def process_voxel(x, y, z, scar_arr, im_arr, thres_values) : 
    scar_value = scar_arr[x, y, z]
    lge_value = im_arr[x, y, z] 

    enhanced_value = scar_value 
    if scar_value > 1:
        enhanced_value = 2
        for th in thres_values :
            enhanced_value += 1 if lge_value > th else 0 

    return enhanced_value
        

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

    threshold_str = list(map(str, args.threshold)) 
    threshold_values = get_threshold_values(args.threshold, mean_bp, std_bp, args.threshold_method)
    output_name = f'enhanced_debug_{"_".join(threshold_str).replace(".","").replace("0", "")}.nii'

    im_array = sitk.GetArrayFromImage(im)
    scar_array = sitk.GetArrayFromImage(scar)
    enhanced_array = np.copy(scar_array)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    voxel_args = []
    for x in range(scar_array.shape[0]):
        for y in range(scar_array.shape[1]):
            for z in range(scar_array.shape[2]): 
                voxel_args.append((x, y, z, scar_array, im_array, threshold_values))
            
    enhanced_values = pool.starmap(process_voxel, voxel_args)
    for idx, (x, y, z) in enumerate(np.ndindex(scar_array.shape)) :
        enhanced_array[x, y, z] = enhanced_values[idx]

    enhanced_scar = sitk.GetImageFromArray(enhanced_array)
    enhanced_scar.SetOrigin(scar.GetOrigin())
    enhanced_scar.SetSpacing(scar.GetSpacing())
    enhanced_scar.SetDirection(scar.GetDirection())

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
