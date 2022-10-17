#!/usr/bin/env python3

import sys
import os
from imatools.vtktools import *
from imatools.ioutils import *

import vtk.util.numpy_support as vtknp

import argparse
import math
import csv
import numpy as np

def compare_fibrosis_per_threshold(msh0, msh1, thresh0, thresh1) : 

    total_area_0 = getSurfaceArea(msh0)
    total_area_1 = getSurfaceArea(msh1)

    if GetMaxScalar(msh0) >= l and GetMaxScalar(msh1) >= l:

        typeThres = "upper" 

        th0 = ugrid2polydata(genericThreshold(msh0, thresh0, typeThres))
        th1 = ugrid2polydata(genericThreshold(msh1, thresh1, typeThres))

        labelscore_0 = getSurfaceArea(th0) / total_area_0
        labelscore_1 = getSurfaceArea(th1) / total_area_1

        # Jaccard is calclated by the Distance points > 1mm
        hd = getHausdorffDistance(th0, th1)

        th_intersect = vtk.vtkThreshold()
        th_intersect.SetInputData(hd)
        th_intersect.SetInputArrayToProcess(
            0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", 'Distance')
        th_intersect.ThresholdByLower(1.0)
        th_intersect.Update()

        intersection = getSurfaceArea(ugrid2polydata(th_intersect.GetOutput()))

        jaccard = intersection / labelscore_0 if (labelscore_0 > 0) else np.inf

    else:
        labelscore_0 = 0
        labelscore_1 = 0
        jaccard = np.inf

        output_list.append(printSurfaceStats(
            entry_A, entry_B0, entry_B1, l, labelscore_0, labelscore_1, jaccard))
