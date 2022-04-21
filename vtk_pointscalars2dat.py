import argparse
import os, sys
import vtk
import numpy
import imatools.ioutils
import imatools.vtktools

inputParser = argparse.ArgumentParser(description="Calculate Hausdorff Distance between 2 meshes")
inputParser.add_argument("baseDir", metavar="path", type=str, help="Directory with data")
inputParser.add_argument("out_dir", metavar="path", type=str, help="Directory with data")
inputParser.add_argument("imsh", metavar="file_name", type=str, help="Mesh name (no ext)")
inputParser.add_argument("oname", metavar="file_name", nargs='?', default="output", type=str, help="Output mesh name")
inputParser.add_argument("scalars_data", metavar="name", nargs='?', default="Distance", type=str, help="Field data name")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

baseDir=args.baseDir
out_dir=args.out_dir
imsh=args.imsh
oname=args.oname
scalars_data=args.scalars_data
verbose=args.verbose

cout("Verbose output", "INFO", verbose)

cout("Base baseDir: "+baseDir, "INPUT", verbose)
cout("Input: " + imsh, "INPUT", verbose)
cout("Output dir: "+out_dir, "INPUT", verbose)
cout("Output: " + oname, "INPUT", verbose)

cout("Loading mesh")
vtko = readVtk(fullfile(baseDir, imsh+".vtk"))

cout("Saving distance pointdata to file", print2console=verbose)
np_distance=convertPointDataToNpArray(vtko, scalars_data)
np.savetxt(fullfile(out_dir, oname+'.dat'), np_distance)
