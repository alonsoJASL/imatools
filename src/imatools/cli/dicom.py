import argparse
import logging
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# patient-table helpers (ported inline from patient_table_from_dicom.py)
# ---------------------------------------------------------------------------


def _extract_patient_info(dicom_file):
    """Read a single DICOM file and return a dict of patient fields."""
    ds = pydicom.dcmread(dicom_file)

    patient_info = {
        "PatientID": ds.PatientID,
        "PatientName": ds.PatientName,
        "PatientAge": ds.PatientAge,
        "PatientSex": ds.PatientSex,
        "PatientWeight": ds.PatientWeight if "PatientWeight" in ds else None,
        "PatientHeight": ds.PatientSize if "PatientSize" in ds else None,
    }

    print(ds)

    return patient_info


def _create_patient_data_table(dicom_dir):
    """Walk *dicom_dir* recursively and build a DataFrame from all .dcm files."""
    logger.info("Processing DICOM files in %s", dicom_dir)
    patient_data = []

    for root, _dirs, files in os.walk(dicom_dir):
        for file in files:
            if file.endswith(".dcm"):
                dicom_file = os.path.join(root, file)
                logger.info("Processing %s", dicom_file)
                patient_info = _extract_patient_info(dicom_file)
                patient_data.append(patient_info)

    df = pd.DataFrame(patient_data)
    return df


# ---------------------------------------------------------------------------
# patient-table handler
# ---------------------------------------------------------------------------


def handle_patient_table(args):
    if args.mode == "folder":
        logger.info("Processing DICOM files in %s", args.input)
        patient_data_table = _create_patient_data_table(args.input)
        logger.info(patient_data_table)
        output_file = args.output
        patient_data_table.to_csv(output_file, index=False)

    elif args.mode == "single":
        logger.info("Processing DICOM file %s", args.input)
        patient_info = _extract_patient_info(args.input)
        logger.info(patient_info)

    return 0


# ---------------------------------------------------------------------------
# stack-slices handler (ported inline from stack_individual_dicoms.py)
# Dead imports resample_to_output / ornt_transform are intentionally omitted.
# ---------------------------------------------------------------------------


def handle_stack_slices(args):
    base_dir = args.dir
    in_folder = args.input_folder
    ofile = args.output

    dicom_directory = os.path.join(base_dir, in_folder)
    dicom_datasets = []
    for filename in os.listdir(dicom_directory):
        if filename.endswith(".dcm"):
            filepath = os.path.join(dicom_directory, filename)
            dataset = pydicom.dcmread(filepath)
            dicom_datasets.append(dataset)

    dicom_datasets.sort(key=lambda x: float(x.SliceLocation))
    volume = np.stack([dataset.pixel_array for dataset in dicom_datasets])
    if args.transpose:
        volume = np.transpose(volume, (1, 2, 0))

    sp_x, sp_y = dicom_datasets[0].PixelSpacing

    spacing_x = sp_x if args.x is None else args.x
    spacing_y = sp_y if args.y is None else args.y
    spacing_z = 1 if args.z is None else args.z

    affine = np.diag([-spacing_x, -spacing_y, spacing_z, 1])
    nifti_image = nib.Nifti1Image(volume, affine)

    output_file = os.path.join(base_dir, ofile)
    nib.save(nifti_image, output_file)

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="imatools-dicom", description="DICOM tools")
    sub = p.add_subparsers(dest="command")

    # patient-table
    pt = sub.add_parser(
        "patient-table",
        help="Extract patient info from DICOM files into a CSV table",
    )
    pt.add_argument(
        "mode",
        choices=["single", "folder"],
        help="Mode of operation: single DICOM file or folder",
    )
    pt.add_argument(
        "-in",
        "--input",
        type=str,
        required=True,
        help="DICOM file (single mode) or directory containing DICOM folders (folder mode)",
    )
    pt.add_argument(
        "-out",
        "--output",
        type=str,
        default="output.csv",
        help="Output CSV file (default: output.csv; folder mode only)",
    )
    pt.set_defaults(func=handle_patient_table)

    # stack-slices
    ss = sub.add_parser(
        "stack-slices",
        help="Stack individual DICOM slices into a NIfTI volume",
    )
    ss.add_argument("-d", "--dir", type=str, required=True, help="Base directory")
    ss.add_argument(
        "-i",
        "--input-folder",
        type=str,
        required=True,
        help="Sub-folder inside --dir containing the .dcm files",
    )
    ss.add_argument(
        "-o",
        "--output",
        type=str,
        default="file.nii.gz",
        help="Output NIfTI filename (relative to --dir; default: file.nii.gz)",
    )
    ss.add_argument(
        "-t",
        "--transpose",
        action="store_true",
        default=False,
        help="Transpose volume axes (1,2,0) after stacking",
    )
    ss.add_argument("-x", "--x", type=float, default=None, help="Override pixel spacing X")
    ss.add_argument("-y", "--y", type=float, default=None, help="Override pixel spacing Y")
    ss.add_argument("-z", "--z", type=float, default=None, help="Override slice spacing Z")
    ss.set_defaults(func=handle_stack_slices)

    return p


def main(args=None):
    parser = _build_parser()
    parsed = parser.parse_args(args)

    if not getattr(parsed, "func", None):
        parser.print_help()
        return 1

    return parsed.func(parsed)


if __name__ == "__main__":
    sys.exit(main())
