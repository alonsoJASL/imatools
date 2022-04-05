import os, sys, subprocess, pdb, re, struct,errno
import vtk
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
