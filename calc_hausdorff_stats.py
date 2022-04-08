import argparse
import os, sys
import vtk
import numpy
from ioutils import *
from vtktools import *

inputParser = argparse.ArgumentParser(description="Load Hausdorff distance mesh, calculate some stats")
inputParser.add_argument("baseDir", metavar="path", type=str, help="Directory with data")
inputParser.add_argument("mshName", metavar="file_name", type=str, help="Mesh name (no ext)")
inputParser.add_argument("outfile", metavar="file_name", nargs='?', default="output", type=str, help="Output mesh name")
inputParser.add_argument("-l", "--labels", metavar="labels", nargs='?', default="1,11,13,15,17,19", type=str, help="Labels (comma-separated) to analyse individually")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

baseDir=args.baseDir
mshName=args.mshName
outfile=args.outfile
labeld=args.labels
verbose=args.verbose
cout("Parsed arguments")

cout("Getting Hausdorff Distance from input")
msh=readVtk(fullfile(baseDir, mshName + '.vtk'))
hd=msh.GetFieldData().GetArray('HausdorffDistance').GetTuple(0)[0]

cout()
pt2cell=vtk.vtkPointDataToCellData()
pt2cell.SetInputData(msh)
pt2cell.Update()

# split labels string into list
cout(labels, "INFO", verbose)

# foreach label:
# threshold msh by label
#
