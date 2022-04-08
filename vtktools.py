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

def getHausdorffDistance(input_mesh0, input_mesh1, label=-1):
    """
    Get Hausdorf Distance between 2 surface meshes
    """
    hd = getHausdorffDistanceFilter(input_mesh0, input_mesh1, label)

    return hd.GetOutput()

def getHausdorffDistanceFilter(input_mesh0, input_mesh1, label=-1):
    """
    Get vtkHausdorffDistancePointSetFilter output between 2 surface meshes
    """
    mesh0 = vtk.vtkPolyData()
    mesh1 = vtk.vtkPolyData()
    if label==-1:
        print("Calculate distance over entire mesh")
        mesh0.DeepCopy(input_mesh0)
        mesh1.DeepCopy(input_mesh1)
    else:
        print("Distance calculated only on label = {}".format(label))
        mesh0=ugrid2polydata(thresholdExactValue(input_mesh0, label))
        mesh1=ugrid2polydata(thresholdExactValue(input_mesh1, label))

    hd=vtk.vtkHausdorffDistancePointSetFilter()
    hd.SetInputData(0, mesh0)
    hd.SetInputData(1, mesh1)
    hd.SetTargetDistanceMethodToPointToCell()
    hd.Update()

    return hd


def thresholdExactValue(msh, exactValue):
    """
    Threshold polydata at exact value (like a tag)
    Returns a unstructured grid
    """
    th=vtk.vtkThreshold()
    th.SetInputData(msh)
    th.ThresholdBetween(exactValue,exactValue)
    th.Update()

    return th.GetOutput()

def convertPointDataToNpArray(vtk_input, str_scalars):
    """
    Convert vtk scalar data to numpy array
    """
    vtkArrayDistance=vtk_input.GetPointData().GetScalars(str_scalars)
    distance=vtk_to_numpy(vtkArrayDistance)

    return distance

def ugrid2polydata(ugrid):
    """
    Converts unstructured grid to polydata using the geometry filter
    """
    gf=vtk.vtkGeometryFilter()
    gf.SetInputData(ugrid)
    gf.Update()

    return gf.GetOutput()
