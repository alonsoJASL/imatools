import sys
sys.path.insert(1, '/Users/jsolislemus/dev/python/quLATi')

from qulati.meshutils import subset_anneal, subset_triangulate, extendMesh
from vtktools import *
from ioutils import *

import argparse
import os, sys
import math
import csv
import numpy as np

inputParser = argparse.ArgumentParser(description="Downsample a mesh")
inputParser.add_argument("base_dir", metavar="base_dir", type=str, help="Directory with data")
inputParser.add_argument("mshname", metavar="msh_name", type=str, help="Mesh name")
inputParser.add_argument("out_dir", metavar="out_dir", nargs='?', default=".",type=str, help="Output directory")
inputParser.add_argument("out_name", metavar="out_name", nargs='?', default="output", type=str, help="Output mesh name")
inputParser.add_argument("-npts", "--num_points", metavar="Num_points_out", nargs='?', default=100, type=int, help="Number of points")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

baseDir=args.base_dir
mshname=args.mshname
out_dir=args.out_dir
out_name=args.out_name
num_points=args.num_points
mode=args.mode
verbose=args.verbose
cout("Parsed arguments", verbose)

cout("Reading input vtk", verbose)
msh=readVtk(fullfile(p2f, mshname+'.vtk'))

pts, el = extractPointsAndElemsFromVtk(msh)
choice = subset_anneal(pts, el, num=num_points, runs=25000)

np.savetxt('test.pts', newPts, header=str(len(newPts)), comments='', fmt='%6.12f')
np.savetxt('test.elem', newEl, header=str(len(newEl)), comments='', fmt='Tr %d %d %d 1')
