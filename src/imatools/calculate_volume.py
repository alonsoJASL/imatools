#!/usr/bin/env python3

from common import vtktools as vtku
from common import ioutils as iou

import vtk 
import argparse

inputParser = argparse.ArgumentParser(description="Get volume and surface area of mesh")
inputParser.add_argument("ipth", metavar="ipth", type=str, help="Full path to mesh")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

ipth=args.ipth
verbose=args.verbose
iou.cout("Parsed arguments", print2console=verbose)

iou.cout("Loading Mesh...", print2console=verbose)
msh=vtku.readVtk(ipth)

iou.cout("Calculating properties...", print2console=verbose)
mp=vtk.vtkMassProperties()
mp.SetInputData(msh)

iou.cout('Area: {} mm^2, Volume: {} mm^3'.format(mp.GetSurfaceArea(), mp.GetVolume()));
