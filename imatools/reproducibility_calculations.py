import argparse
import os, sys
import vtk
import numpy
from imatools.ioutils import *
from imatools.vtktools import *

import csv

def printSurfaceStats(a,b0,b1,l, area0, area1, jacc):
    return [a, b0+b1, l, area0, area1, jacc]

def printHausdorffStats(a,b0,b1, lab, hd):
    hd_aux=hd.GetHausdorffDistance()
    dist=convertPointDataToNpArray(hd.GetOutput(), 'Distance')

    return [a, b0+b1, lab, hd.GetHausdorffDistance(), np.median(dist), np.mean(dist), np.std(dist)]
