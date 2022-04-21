import argparse
import os, sys
import vtk
import numpy
import imatools.ioutils
import imatools.vtktools

import csv

def extractHausdorffStats(hd, u, p, lab=0):
    hd_aux=hd.GetHausdorffDistance()
    dist=convertPointDataToNpArray(hd.GetOutput(), 'Distance')

    return [u, p, lab, hd.GetHausdorffDistance(), np.median(dist), np.mean(dist), np.std(dist)]


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
print(labels)

intra_cases=readFileToList(comparisons_file)
output_list = [['user', 'case', 'label', 'Hausdorff', 'median', 'mean', 'stdev'],]

for comp in intra_cases:
    user=comp[0].strip()
    patient0=comp[1].strip()
    patient1=comp[2].strip()

    patient = patient0[0:-2]
    cout("USER: {}, PATIENT: {}".format(user, patient))

    l0=searchFileByType(fullfile(baseDir, user, '03_completed', patient0), msh_prefix, 'vtk')
    l1=searchFileByType(fullfile(baseDir, user, '03_completed', patient1), msh_prefix, 'vtk')

    msh0 = readVtk(l0[0])
    msh1 = readVtk(l1[0])


    cout(patient, "SAVE", save_meshes and verbose)
    if save_meshes and not os.path.isdir(fullfile(out_dir, patient)):
        os.mkdir(fullfile(out_dir, patient))

    for l in labels:
        hdl=getHausdorffDistanceFilter(msh0, msh1, l, verbose)
        output_list.append(extractHausdorffStats(hdl, user, patient, l))

        if save_meshes:
            omsh = "{}_Hausdorff".format(user[0:3]) if (l==0) else "{}_thresholded_{}".format(user[0:3], l)
            writeVtk(hdl.GetOutput(), fullfile(out_dir, patient), omsh)

out_path = fullfile(out_dir, out_name)
savetype = 'a' if os.path.exists(out_path) else 'w'
cout("{} to file".format("APPEND" if savetype=='a' else "WRITE"), print2console=verbose)

with open(out_path, savetype) as f:
    writer = csv.writer(f)
    writer.writerows(output_list)
    # for row in output_list:
    #     writer.writerow(row)
