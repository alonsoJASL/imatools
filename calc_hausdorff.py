import argparse
import os, sys
import vtk
import numpy
from vtk.util.numpy_support import vtk_to_numpy
from ioutils import *
from vtktools import *

inputParser = argparse.ArgumentParser(description="Calculate Hausdorff Distance between 2 meshes")
inputParser.add_argument("dir_0", metavar="path", type=str, help="Directory with data")
inputParser.add_argument("dir_1", metavar="path", type=str, help="Directory with data")
inputParser.add_argument("dir_out", metavar="path", type=str, help="Directory with data")
inputParser.add_argument("mesh_0", metavar="file_name", type=str, help="Mesh name (no ext)")
inputParser.add_argument("mesh_1", metavar="file_name", type=str, help="Mesh name (no ext)")
inputParser.add_argument("mesh_out", metavar="file_name", nargs='?', default="output", type=str, help="Output mesh name")

args = inputParser.parse_args()

dir_0=args.dir_0
dir_1=args.dir_1
dir_out=args.dir_out
mesh_0=args.mesh_0
mesh_1=args.mesh_1
mesh_out=args.mesh_out

cout("Base dir_0: "+dir_0, "INPUT")
cout("Base dir_1: "+dir_1, "INPUT")
cout("Input 0 " + mesh_0, "INPUT")
cout("Input 1 " + mesh_1, "INPUT")
cout("Output dir_out: "+dir_out, "INPUT")
cout("Output " + mesh_out, "INPUT")

cout("Loading meshes")
vtk0 = readVtk(fullfile(dir_0, mesh_0+".vtk"))
vtk1 = readVtk(fullfile(dir_1, mesh_1+".vtk"))

cout("Calculating Hausdorff distance")
vtko = getHausdorfDistance(vtk0, vtk1)

cout("Writing {} mesh".format(mesh_out))
writeVtk(vtko, dir_out, mesh_out)

cout("Saving distance pointdata to file")
vtkArrayDistance=vtko.GetPointData().GetScalars("Distance")
distance=vtk_to_numpy(vtkArrayDistance)
np.savetxt(fullfile(dir_out, mesh_out+'.dat'), np_distance)
