import os
import argparse
import pandas as pd
import pydicom

from common.config import configure_logging

# Configure logging
logger = configure_logging(__name__)

def extract_patient_info(dicom_file):
    # Load DICOM file
    ds = pydicom.dcmread(dicom_file)
    
    # Extract patient information
    patient_info = {
        'PatientID': ds.PatientID,
        'PatientName': ds.PatientName,
        'PatientAge': ds.PatientAge,
        'PatientSex': ds.PatientSex,
        'PatientWeight': ds.PatientWeight if 'PatientWeight' in ds else None,
        'PatientHeight': ds.PatientSize if 'PatientSize' in ds else None
    }

    print(ds)
    
    return patient_info

def create_patient_data_table(dicom_dir):
    # Initialize an empty list to store patient information
    logger.info(f'Processing DICOM files in {dicom_dir}')
    patient_data = []
    
    # Iterate through DICOM files in the directory
    for root, dirs, files in os.walk(dicom_dir):
        for file in files:
            if file.endswith('.dcm'):
                dicom_file = os.path.join(root, file)
                logger.info(f'Processing {dicom_file}')
                patient_info = extract_patient_info(dicom_file)
                patient_data.append(patient_info)
    
    # Create a DataFrame from the patient data list
    df = pd.DataFrame(patient_data)
    
    return df

def process_folder(args) : 
    # Directory containing DICOM files
    dicom_dir = args.input 

    # Create patient data table
    patient_data_table = create_patient_data_table(dicom_dir)

    # Display the data table
    logger.info(patient_data_table)

    # Save the data table to a CSV file
    output_file = args.output
    patient_data_table.to_csv(output_file, index=False)

def process_single(args) :
    dicom_file = args.input
    patient_info = extract_patient_info(dicom_file)
    logger.info(patient_info)

def main(args) : 
    if args.mode == 'folder' : 
        logger.info(f'Processing DICOM files in {args.input}')
        process_folder(args)
    
    elif args.mode == 'single' :
        logger.info(f'Processing DICOM file {args.input}')

if __name__ == '__main__':
    input_parser = argparse.ArgumentParser(description='Create a patient data table from DICOM files')
    input_parser.add_argument('mode', choices=['single', 'folder'], help='Mode of operation')
    input_parser.add_argument('-in', '--input', type=str, help='Directory containing DICOM folders')
    input_parser.add_argument('-out', '--output', type=str, default='output.csv', help='Output CSV file')

    args = input_parser.parse_args()
    main(args)



