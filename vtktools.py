import os, sys, subprocess, pdb, re, struct,errno
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

def readVtk(fname):
    """
    Read VTK file
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()

    return reader.GetOutput()

def writeVtk(mesh, directory, outname="output"):
    writer=vtk.vtkPolyDataWriter()
    writer.SetInputData(mesh)
    writer.SetFileName(directory+"/"+outname+".vtk")
    writer.SetFileTypeToASCII()
    writer.Update()

def getHausdorfDistance(mesh0, mesh1):
    """
    Get Hausdorf Distance between 2 surface meshes
    """
    hd=vtk.vtkHausdorffDistancePointSetFilter()
    hd.SetInputData(0, mesh0)
    hd.SetInputData(1, mesh1)
    hd.SetTargetDistanceMethodToPointToCell()
    hd.Update()

    return hd.GetOutput()

def convertPointDataToNpArray(vtk_input, str_scalars):
    """
    Convert vtk scalar data to numpy array
    """
    vtkArrayDistance=vtk_input.GetPointData().GetScalars(str_scalars)
    distance=vtk_to_numpy(vtkArrayDistance)

    return distance
