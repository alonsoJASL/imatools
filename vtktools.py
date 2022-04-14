import os, sys, subprocess, pdb, re, struct,errno
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from ioutils import cout
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

def getHausdorffDistance(input_mesh0, input_mesh1, label=0):
    """
    Get Hausdorf Distance between 2 surface meshes
    """
    hd = getHausdorffDistanceFilter(input_mesh0, input_mesh1, label)

    return hd.GetOutput()

def getHausdorffDistanceFilter(input_mesh0, input_mesh1, label=0, verbose=False):
    """
    Get vtkHausdorffDistancePointSetFilter output between 2 surface meshes
    """
    mesh0 = vtk.vtkPolyData()
    mesh1 = vtk.vtkPolyData()
    if label==0:
        cout("Calculate distance over entire mesh", print2console=verbose)
        mesh0.DeepCopy(input_mesh0)
        mesh1.DeepCopy(input_mesh1)
    else:
        cout("Distance calculated only on label = {}".format(label), print2console=verbose)
        mesh0=ugrid2polydata(thresholdExactValue(input_mesh0, label))
        mesh1=ugrid2polydata(thresholdExactValue(input_mesh1, label))

    hd=vtk.vtkHausdorffDistancePointSetFilter()
    hd.SetInputData(0, mesh0)
    hd.SetInputData(1, mesh1)
    hd.SetTargetDistanceMethodToPointToCell()
    hd.Update()

    return hd

def genericThreshold(msh, exactValue, typeThres='exact', verbose=False):
    """
    Threshold polydata
    Returns a unstructured grid
    """
    cout("Threshold type: {}".format(typeThres), "genericThreshold", verbose)
    thresBehaviour={'exact': 0, 'upper': 1, 'lower': 2}

    th=vtk.vtkThreshold()
    th.SetInputData(msh)

    if thresBehaviour[typeThres] == 0: # exact
        th.ThresholdBetween(exactValue,exactValue)
    elif thresBehaviour[typeThres] == 1: # upper
        th.ThresholdByUpper(exactValue)
    elif thresBehaviour[typeThres] == 2: # lower
        th.ThresholdByLower(exactValue)
    else:
        cout("Wrong type of threshold", "ERROR")
        sys.exit(-1)

    th.Update()

    return th.GetOutput()

def thresholdExactValue(msh, exactValue):
    """
    Threshold polydata at exact value (like a tag)
    Returns a unstructured grid
    """

    return genericThreshold(msh, exactValue, 'exact')

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

def getSurfaceArea(msh):
    mp = vtk.vtkMassProperties();
    mp.SetInputData(msh)
    return mp.GetSurfaceArea()

def getBooleanOperationFilter(msh0, msh1, operation_str='union'):
    opts={'union': 0, 'intersection': 1, 'difference': 2}
    bopd = vtk.vtkBooleanOperationPolyDataFilter()
    bopd.SetOperation(opts[operation_str])
    bopd.SetInputData(0, msh0)
    bopd.SetInputData(1, msh1)
    bopd.Update()

    return bopd

def getBooleanOperation(msh0, msh1, operation_str='union'):
    bopd = getBooleanOperationFilter(msh0, msh1, operation_str)
    return bopd.GetOutput()

def getSurfacesJaccard(pd1, pd2):
    """
    computes the jaccard distance for two polydata objects
    """
    union = getBooleanOperation(pd1, pd2, 'union')
    intersection = getBooleanOperation(pd1, pd2, 'intersection')

    return getSurfaceArea(intersection)/getSurfaceArea(union)
