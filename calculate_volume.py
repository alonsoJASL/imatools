#!/usr/bin/env python3

import sys, os
from imatools.vtktools import *
from imatools.ioutils import *

import vtk.util.numpy_support as vtknp

import argparse
import math
import csv
import numpy as np

inputParser = argparse.ArgumentParser(description="Get volume and surface area of mesh")
inputParser.add_argument("ipth", metavar="ipth", type=str, help="Full path to mesh")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

ipth=args.ipth
verbose=args.verbose
cout("Parsed arguments", print2console=verbose)

cout("Loading Mesh...", print2console=verbose)
msh=readVtk(ipth)

cout("Calculating properties...", print2console=verbose)
mp=vtk.vtkMassProperties()
mp.SetInputData(msh)

cout('Area: {} mm^2, Volume: {} mm^3'.format(mp.GetSurfaceArea(), mp.GetVolume()));
