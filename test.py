import argparse
import os, sys
import vtk
import numpy
from ioutils import *
from vtktools import *

inputParser = argparse.ArgumentParser(description="Various tests")
inputParser.add_argument("directory", metavar="path", type=str, help="Directory with data")
inputParser.add_argument("mesh_0", metavar="file_name", type=str, help="Mesh name (no ext)")
inputParser.add_argument("mesh_1", metavar="file_name", type=str, help="Mesh name (no ext)")
inputParser.add_argument("mesh_out", metavar="file_name", type=str, help="Output mesh name")

args = inputParser.parse_args()

directory=args.directory
mesh_0=args.mesh_0
mesh_1=args.mesh_1
mesh_out=args.mesh_out

print("Base directory: "+directory)
print("Input 0 " + mesh_0)
print("Input 1 " + mesh_1)
print("Output " + mesh_out)

# ptsname = fullfile(directory, mesh_0+'.pts')
# elemname = fullfile(directory, mesh_0 + ".elem")
# pts, nPts = readParsePts(ptsname)
# el, nElem = readParseElem(elemname)
#
# print("Name: {}\n Number of nodes: {}".format(ptsname, nPts))
# print(pts)
# print("Number of elements: {}".format(nElem))
# print(el[1:10])

vtk0 = readVtk(fullfile(directory, mesh_0+".vtk"))
vtk1 = readVtk(fullfile(directory, mesh_1+".vtk"))

vtko = getHausdorfDistance(vtk0, vtk1)

writeVtk(vtko, directory, mesh_out)
