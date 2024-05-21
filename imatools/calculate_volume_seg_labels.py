#!/usr/bin/env python3

from common import ioutils as iou
from common import itktools
from common.config import configure_logging	

import SimpleITK as sitk 
import os
import argparse
import numpy as np

logger = configure_logging(__name__)

def volume_message(label_indx, volume_dict: dict, units='mL') : 
	multiplier = 1e-3 if units == 'mL' else 1
	unit_str = 'mL' if units == 'mL' else 'mm^3'
	mystr = f"Label {label_indx}: {np.round(volume_dict[label_indx] * multiplier, 3)} {unit_str}"

	return mystr

def main(args) : 
	ipth=args.input
	base_dir=os.path.dirname(ipth)
	units=args.units
	verbose=args.verbose
	iou.cout("Parsed arguments", print2console=verbose, logger=logger)

	iou.cout("Loading segmentation...", print2console=verbose, logger=logger)
	im=sitk.ReadImage(ipth)

	iou.cout("Calculating volumes...", print2console=verbose, logger=logger)
	vols=itktools.get_labels_volumes(im)

	iou.cout('Assuming that the image is in mm.\n', logger=logger)
	if args.output == '' :
		for i in sorted(vols.keys()):
			iou.cout(volume_message(i, vols, units), logger=logger)
	else :
		with open(os.path.join(base_dir, args.output), 'w') as f:
			for i in sorted(vols.keys()):
				f.write(volume_message(i, vols, units))

	# for i in sorted(vols.keys()):
	# 	iou.cout(f"Label {i}: {np.round(vols[i]*1e-3,3)} mL\n")



if __name__ == "__main__":
	inputParser = argparse.ArgumentParser(description="Get volumes of all the labels of a segmentation")
	inputParser.add_argument("--input", "-in", metavar="ipth", type=str, help="Full path to segmentation")
	inputParser.add_argument("--output", '-out', metavar="out_name", type=str, default='', help="Output file name")
	inputParser.add_argument("--units", "-units", choices=['mm', 'mL'], default='mL', help='Units of the output volumes (default: mL)')
	inputParser.add_argument("--verbose", "-v", action='store_true', help="Verbose output")

	args = inputParser.parse_args()

	main(args)
