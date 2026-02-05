import os
import argparse
import numpy as np 
import pydicom 
import nibabel as nib

from nibabel.processing import resample_to_output
from nibabel.orientations import ornt_transform

def main(args) : 

    base_dir = args.dir
    in_folder = args.input_folder
    ofile = args.output

    dicom_directory = f'{base_dir}/{in_folder}'
    dicom_datasets = []
    for filename in os.listdir(dicom_directory):
        if filename.endswith(".dcm"):
            filepath = os.path.join(dicom_directory, filename)
            dataset = pydicom.dcmread(filepath)
            dicom_datasets.append(dataset)
    
    dicom_datasets.sort(key=lambda x: float(x.SliceLocation))
    volume = np.stack([dataset.pixel_array for dataset in dicom_datasets])
    if args.transpose :
        volume = np.transpose(volume, (1,2,0))

    sp_x, sp_y = dicom_datasets[0].PixelSpacing

    spacing_x = sp_x if args.x is None else args.x
    spacing_y = sp_y if args.y is None else args.y
    spacing_z = 1 if args.z is None else args.z


    affine = np.diag([-spacing_x, -spacing_y, spacing_z, 1])
    nifti_image = nib.Nifti1Image(volume, affine)

    output_file = f'{base_dir}/{ofile}'
    nib.save(nifti_image, output_file)

if __name__ == '__main__':
    in_parse = argparse.ArgumentParser()
    in_parse.add_argument('-d', '--dir', type=str, required=True)
    in_parse.add_argument('-i', '--input-folder', type=str, required=True)
    in_parse.add_argument('-o', '--output', type=str, required=False, default='file.nii.gz')
    in_parse.add_argument('-t', '--transpose', action='store_true', default=False)
    in_parse.add_argument('-x', '--x', type=float, required=False)
    in_parse.add_argument('-y', '--y', type=float, required=False)
    in_parse.add_argument('-z', '--z', type=float, required=False)

    args = in_parse.parse_args()
    main(args)