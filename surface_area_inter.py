import argparse
import os, sys
import math
import vtk
import csv
import numpy as np
import pandas as pd

from ioutils import *
from vtktools import *

import plotutils as myplot

def surfaceStats(a,b0,b1,l, area0, area1, jacc):
    return [a, b0+b1, l, area0, area1, jacc]

inputParser = argparse.ArgumentParser(description="Load Hausdorff distance mesh, calculate some stats")
inputParser.add_argument("base_dir", metavar="base_dir", type=str, help="Directory with data")
inputParser.add_argument("comparisons_file", metavar="comparisons_file", type=str, help="File with inter comparisons")
inputParser.add_argument("msh_prefix", metavar="msh_name_prefix", nargs='?', default="clean", type=str, help="Comparison mesh prefix (default=clean)")
inputParser.add_argument("out_dir", metavar="out_dir", nargs='?', default=".",type=str, help="Output directory")
inputParser.add_argument("out_name", metavar="out_name", nargs='?', default="intra.csv", type=str, help="Output csv name (saved on out_dir)")
inputParser.add_argument("-l", "--labels", metavar="labels", nargs='?', default="1,11,13,15,17,19", type=str, help="Labels (comma-separated, 0=all)")
inputParser.add_argument("-m", "--mode", metavar="mode", nargs='?', default=0, type=int, help="Mode (0=inter, 1=intra)")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

baseDir=args.base_dir
comparisons_file=args.comparisons_file
msh_prefix=args.msh_prefix
out_dir=args.out_dir
out_name=args.out_name
labels_str=args.labels
mode=args.mode
verbose=args.verbose
cout("Parsed arguments")

cout(labels_str, "LABELS", verbose)
thresTypeScar = ("Scar" in msh_prefix)
cout("Computing scar threshold", print2console=thresTypeScar)

if thresTypeScar:
    labels = [float(n.strip()) for n in labels_str.split(',')]
else:
    labels = [int(n.strip()) for n in labels_str.split(',')]

if mode>1:
    cout("Modes supported (-m MODE): INTER=0, INTRA=1", "ERROR")
    sys.exit(-1)


file_cases=readFileToList(comparisons_file)
output_list = [['case','comparison','label','area_0','area_1','jaccard'],]
# case:         patient / user
# comparison:   u0, u1 / p0, p1
# label:        tag or threshold
# area_0:       A(u0) / A(p0)
# area_1:       A(u1) / A(p1)
# jaccard:      intersection(0,1)/union(0,1)

# for each label/threshold
for comp in file_cases:
    # MODE 0 (inter): entry_A=patient, entry_B0=user0, entry_B1=user1
    # MODE 1 (intra): a=user, entry_B0=patient0, entry_B1=patient1
    entry_A=comp[0].strip()
    entry_B0=comp[1].strip()
    entry_B1=comp[2].strip()

    if mode==0:
        p0=entry_A
        p1=entry_A
        u0=entry_B0
        u1=entry_B1
    else:
        p0=entry_B0
        p1=entry_B1
        u0=entry_A
        u1=entry_A

    patient = entry_A if(mode==0) else entry_B0[0:-2]

    path0 = fullfile(baseDir, u0, '03_completed', p0)
    path1 = fullfile(baseDir, u1, '03_completed', p1)

    cout(path0)
    cout(path1)

    l0=searchFileByType(path0, msh_prefix, 'vtk')
    l1=searchFileByType(path1, msh_prefix, 'vtk')

    msh0=readVtk(l0[0])
    msh1=readVtk(l1[0])

    cout("Processing labels ({},{}) vs ({},{})".format(u0, p0, u1, p1))
    cout(path0, 'Path_0')
    cout(path1, 'Path_1')
    for l in labels:
        cout(l, 'LABELS', verbose)
        typeThres = "upper" if (thresTypeScar) else "exact"
        th0 = ugrid2polydata(genericThreshold(msh0, l, typeThres))
        th1 = ugrid2polydata(genericThreshold(msh1, l, typeThres))

        area_0 = getSurfaceArea(th0)
        area_1 = getSurfaceArea(th1)

        # Jaccard is calclated by the Distance points > 1mm
        hd = getHausdorffDistance(th0, th1)

        th_intersect=vtk.vtkThreshold()
        th_intersect.SetInputData(hd)
        th_intersect.SetInputArrayToProcess(0,0,0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", 'Distance')
        th_intersect.ThresholdByLower(1.0)
        th_intersect.Update()

        intersection = getSurfaceArea(ugrid2polydata(th_intersect.GetOutput()))

        jaccard = intersection/area_0

        output_list.append(surfaceStats(entry_A, entry_B0, entry_B1, l, area_0, area_1, jaccard))


with open(fullfile(out_dir, out_name), "w") as f:
    writer = csv.writer(f)
    writer.writerows(output_list)
