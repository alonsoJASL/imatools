import argparse
import os, sys
import vtk
import numpy
from imatools.ioutils import *
from imatools.vtktools import *

import csv

def extractHausdorffStats(hd, p, lab=-1):
    hd_aux=hd.GetHausdorffDistance()
    dist=convertPointDataToNpArray(hd.GetOutput(), 'Distance')

    return [p, lab, hd.GetHausdorffDistance(), np.median(dist), np.mean(dist), np.std(dist)]


inputParser = argparse.ArgumentParser(description="Load Hausdorff distance mesh, calculate some stats")
inputParser.add_argument("base_dir", metavar="path", type=str, help="Directory with data")
inputParser.add_argument("comparisons_file", metavar="file_name", type=str, help="File with inter comparisons")
inputParser.add_argument("msh_prefix", metavar="name", nargs='?', default="clean", type=str, help="Comparison mesh prefix (default=clean)")
inputParser.add_argument("out_dir", metavar="path", nargs='?', default=".",type=str, help="Output directory")
inputParser.add_argument("out_name", metavar="file_name", nargs='?', default="inter.csv", type=str, help="Output csv name (saved on out_dir)")
inputParser.add_argument("-l", "--labels", metavar="labels", nargs='?', default="1,11,13,15,17,19", type=str, help="Labels (comma-separated, -1=all)")
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
output_list = [['case', 'label', 'Hausdorff', 'median', 'mean', 'stdev'],]

for comp in inter_cases:
    patient=comp[0].strip()
    u0=comp[1].strip()
    u1=comp[2].strip()

    l0=searchFileByType(fullfile(baseDir, u0, '03_completed', patient), msh_prefix, 'vtk')
    l1=searchFileByType(fullfile(baseDir, u1, '03_completed', patient), msh_prefix, 'vtk')

    msh0 = readVtk(l0[0])
    msh1 = readVtk(l1[0])

    cout(patient, "SAVE", save_meshes)
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
    # for row in output_list:
    #     writer.writerow(row)
