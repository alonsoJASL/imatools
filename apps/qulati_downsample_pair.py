import sys, os
QULATI_DIR=os.getcwd()+'/../quLATi'
sys.path.insert(1, QULATI_DIR)

from qulati.meshutils import subset_anneal, subset_triangulate, extendMesh
from imatools.vtktools import *
from imatools.ioutils import *

import vtk.util.numpy_support as vtknp

import argparse
import math
import csv
import numpy as np

inputParser = argparse.ArgumentParser(description="Downsample a mesh")
inputParser.add_argument("base_dir", metavar="base_dir", type=str, help="Directory with data")
inputParser.add_argument("mshname1", metavar="msh_name1", type=str, help="Mesh name")
inputParser.add_argument("mshname2", metavar="msh_name2", type=str, help="Mesh name")
inputParser.add_argument("-carp", "--save2carp", action='store_true', help="Save output to carp")
inputParser.add_argument("-vtk", "--save2vtk", action='store_true', help="Save output to carp")
inputParser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")

args = inputParser.parse_args()

baseDir=args.base_dir
mshname1=args.mshname1
mshname2=args.mshname2
save2carp=args.save2carp
save2vtk=args.save2vtk
verbose=args.verbose
cout("Parsed arguments", print2console=verbose)

msh1=readVtk(fullfile(baseDir, mshname1))
msh2=readVtk(fullfile(baseDir, mshname2))
n1=msh1.GetNumberOfPoints()
n2=msh2.GetNumberOfPoints()

if n1>n2:
    largeMsh=msh1
    largeN=n1
    smallN=n2
    outLargeName=mshname1
elif n1<n2:
    largeMsh=msh2
    largeN=n2
    smallN=n1
    outLargeName=mshname2
else:
    cout("Same sizes: cancelling operation", 'ATTENTION')
    sys.exit(0)

outLargeName=outLargeName[0:-4]

cout("Downsampling {} from {} to {} points".format(outLargeName, largeN, smallN), print2console=verbose)
largePts, largeEl = extractPointsAndElemsFromVtk(largeMsh)
if largeMsh.GetPointData().GetScalars() is None :
    cout('Attempting to pass cell data to point data in {}'.format(outLargeName), print2console=verbose)
    c2p=vtk.vtkCellDataToPointData()
    c2p.SetInputData(largeMsh)
    c2p.Update()
    largeMsh=c2p.GetOutput()

largeScar=convertPointDataToNpArray(largeMsh, 'scalars')

choice=subset_anneal(largePts, largeEl, num=smallN, runs=3000)
newPts, newEl = subset_triangulate(largePts, largeEl, choice, holes=5)
dat=largeScar[choice]

if save2vtk:
    cout("Save to vtk", print2console=verbose)
    nodes = vtk.vtkPoints()
    for ix in range(len(newPts)):
        nodes.InsertPoint(ix, newPts[ix,0], newPts[ix,1], newPts[ix,2])

    elems = vtk.vtkCellArray()
    for ix in range(len(newEl)):
        elIdList=vtk.vtkIdList()
        elIdList.InsertNextId(newEl[ix,0])
        elIdList.InsertNextId(newEl[ix,1])
        elIdList.InsertNextId(newEl[ix,2])
        elems.InsertNextCell(elIdList)

    pd=vtk.vtkPolyData()
    pd.SetPoints(nodes)
    pd.SetPolys(elems)
    pd.GetPointData().SetScalars(vtknp.numpy_to_vtk(dat))
    p2c=vtk.vtkPointDataToCellData()
    p2c.SetInputData(pd)
    p2c.Update()
    writeVtk(p2c.GetOutput(), baseDir, "downsample_"+outLargeName)


# save to carp
if save2carp:
    cout("Save to vtk", print2console=verbose)
    saveToCarpTxt(newPts, newEl, fullfile(baseDir,"downsample_"+outLargeName))
    np.savetxt(fullfile(baseDir, "downsample_"+outLargeName+'.dat'), dat)
