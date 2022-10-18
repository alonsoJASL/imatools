import sys, os

from imatools.ioutils import *
from imatools.vtktools import *
import argparse

"""
Loads a mesh
"""

inputParser = argparse.ArgumentParser(description="Test on ioutils")
inputParser.add_argument("base_dir", metavar="base_dir", type=str, help="Directory with data")
inputParser.add_argument("mshname1", metavar="msh_name1", type=str, help="Mesh name")

args = inputParser.parse_args()

baseDir=args.base_dir
mshname1=args.mshname1

cout("Parsed Arguments")
pd=vtk.vtkPolyData()
print(pd)
