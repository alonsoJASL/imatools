import argparse
import os, sys
import math
import vtk
import numpy as np
import pandas as pd

from ioutils import *
from vtktools import *

import plotutils as myplot

inputParser = argparse.ArgumentParser(description="Load Hausdorff distance mesh, calculate some stats")
inputParser.add_argument("base_dir", metavar="base_dir", type=str, help="Directory with data")
inputParser.add_argument("comparisons_file", metavar="comparisons_file", type=str, help="File with inter comparisons")
inputParser.add_argument("msh_prefix", metavar="msh_name_prefix", nargs='?', default="clean", type=str, help="Comparison mesh prefix (default=clean)")
inputParser.add_argument("out_dir", metavar="out_dir", nargs='?', default=".",type=str, help="Output directory")
inputParser.add_argument("out_name", metavar="out_name", nargs='?', default="intra.csv", type=str, help="Output csv name (saved on out_dir)")
inputParser.add_argument("-l", "--labels", metavar="labels", nargs='?', default="1,11,13,15,17,19", type=str, help="Labels (comma-separated, 0=all)")
inputParser.add_argument("-s", "--save", action='store_true', help="Save outputs")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

baseDir=args.base_dir
comparisons_file=args.comparisons_file
msh_prefix=args.msh_prefix
out_dir=args.out_dir
out_name=args.out_name
labels_str=args.labels
save_meshes=args.save
verbose=args.verbose
cout("Parsed arguments")

cout(labels_str, "LABELS", verbose)
labels = [int(n.strip()) for n in labels_str.split(',')]

inter_cases=readFileToList(comparisons_file)
# output_list = [['case', 'label', 'Hausdorff', 'median', 'mean', 'stdev'],]

# for each label/threshold
# threshold the


# START HERE
