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
inputParser.add_argument("-m", "--mode", metavar="mode", nargs='?', default=0, type=int, help="Mode (0=inter, 1=intra)")
inputParser.add_argument("-s", "--save", action='store_true', help="Save outputs")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

baseDir=args.base_dir
comparisons_file=args.comparisons_file
msh_prefix=args.msh_prefix
out_dir=args.out_dir
out_name=args.out_name
labels_str=args.labels
mode=args.mode
save_meshes=args.save
verbose=args.verbose
cout("Parsed arguments")

cout(labels_str, "LABELS", verbose)
labels = [int(n.strip()) for n in labels_str.split(',')]

if mode>1:
    cout("Modes supported (-m MODE): INTER=0, INTRA=1", "ERROR")
    sys.exit(-1)

file_cases=readFileToList(comparisons_file)
if mode==0:
    output_list = [['patient', 'user0', 'user1','label', 'Hausdorff', 'median', 'mean', 'stdev'],]
else:
    output_list = [['patient', 'user', 'label', 'Hausdorff', 'median', 'mean', 'stdev'],]

# for each label/threshold
for comp in file_cases:
    # MODE 0 (inter): a=patient, b0=user0, b1=user1
    # MODE 1 (intra): a=user, b0=patient0, b1=patient1
    a=comp[0].strip()
    b0=comp[1].strip()
    b1=comp[2].strip()

    if mode==0:
        p0=a
        p1=a
        u0=b0
        u1=b1
    else:
        p0=b0
        p1=b1
        u0=a
        u1=a

    patient = a if(mode==0) else b0[0:-2]

    path0 = fullfile(baseDir, u0, '03_completed', p0)
    path1 = fullfile(baseDir, u1, '03_completed', p1)

    l0=searchFileByType(path0, msh_prefix, 'vtk')
    l1=searchFileByType(path1, msh_prefix, 'vtk')

    msh0=readVtk(l0[0])
    msh1=readVtk(l1[0])

    cout(patient, "SAVE", save_meshes and verbose)
    if save_meshes and not os.path.isdir(fullfile(out_dir, patient)):
        os.mkdir(fullfile(out_dir, patient))

    for l in labels:
        hdl=getHausdorffDistanceFilter(msh0, msh1, l)
        output_list.append(extractHausdorffStats(hdl, patient, l))

        if save_meshes:
            omsh = "{}_Hausdorff".format(patient) if (l==0) else "thresholded_{}".format(l)
            writeVtk(hdl.GetOutput(), fullfile(out_dir, patient), omsh)

with open(fullfile(out_dir, out_name), "w") as f:
    writer = csv.writer(f)
    writer.writerows(output_list)
