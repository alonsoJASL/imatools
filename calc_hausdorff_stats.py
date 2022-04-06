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
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

baseDir=args.baseDir
mshName=args.mshName
outfile=args.outfile
verbose=args.verbose

msh=readVtk(fullfile(baseDir, mshName + '.vtk'))
hd=msh.GetFieldData().GetArray('HausdorffDistance').GetTuple(0)[0]

pt2cell=vtk.vtkPointDataToCellData()
pt2cell.SetInputData(msh)
pt2cell.Update()
