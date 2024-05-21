#!/usr/bin/env python3

from common import ioutils as iou
from common import itktools

import SimpleITK as sitk 
import argparse
import numpy as np

inputParser = argparse.ArgumentParser(description="Get volumes of all the labels of a segmentation")
inputParser.add_argument("ipth", metavar="ipth", type=str, help="Full path to segmentation")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

ipth=args.ipth
verbose=args.verbose
iou.cout("Parsed arguments", print2console=verbose)

iou.cout("Loading segmentation...", print2console=verbose)
im=sitk.ReadImage(ipth)

iou.cout("Calculating volumes...", print2console=verbose)
vols=itktools.get_labels_volumes(im)

iou.cout('Assuming that the image is in mm.\n')
for i in sorted(vols.keys()):
	iou.cout(f"Label {i}: {np.round(vols[i]*1e-3,3)} mL\n")