raise ImportError(
    "imatools.qulati_downsample_pair is parked as quLATi is not yet "
    "packaged as an optional dependency. See future_work.md."
)


import os
import sys

QULATI_DIR = os.getcwd() + "/../quLATi"
sys.path.insert(1, QULATI_DIR)

import argparse

import numpy as np
import vtk
import vtk.util.numpy_support as vtknp
from common import ioutils as iou
from common import vtktools as vtku
from qulati.meshutils import subset_anneal, subset_triangulate

parser = argparse.ArgumentParser(description="Downsample a mesh")
parser.add_argument("base_dir", metavar="base_dir", type=str, help="Directory with data")
parser.add_argument("mshname1", metavar="msh_name1", type=str, help="Mesh name")
parser.add_argument("mshname2", metavar="msh_name2", type=str, help="Mesh name")
parser.add_argument("-carp", "--save2carp", action="store_true", help="Save output to carp")
parser.add_argument("-vtk", "--save2vtk", action="store_true", help="Save output to carp")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

args = parser.parse_args()

base_dir = args.base_dir
mshname1 = args.mshname1
mshname2 = args.mshname2
save2carp = args.save2carp
save2vtk = args.save2vtk
verbose = args.verbose
iou.cout("Parsed arguments", print2console=verbose)

msh1 = vtku.readVtk(iou.fullfile(base_dir, mshname1))
msh2 = vtku.readVtk(iou.fullfile(base_dir, mshname2))
n1 = msh1.GetNumberOfPoints()
n2 = msh2.GetNumberOfPoints()

if n1 > n2:
    large_mesh = msh1
    large_size = n1
    small_size = n2
    out_large_name = mshname1
elif n1 < n2:
    large_mesh = msh2
    large_size = n2
    small_size = n1
    out_large_name = mshname2
else:
    iou.cout("Same sizes: cancelling operation", "ATTENTION")
    sys.exit(0)

out_large_name = out_large_name[0:-4]

iou.cout(
    "Downsampling {} from {} to {} points".format(out_large_name, large_size, small_size),
    print2console=verbose,
)
large_pts, large_el = vtku.extractPointsAndElemsFromVtk(large_mesh)
if large_mesh.GetPointData().GetScalars() is None:
    iou.cout(
        "Attempting to pass cell data to point data in {}".format(out_large_name),
        print2console=verbose,
    )
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(large_mesh)
    c2p.Update()
    large_mesh = c2p.GetOutput()

large_scar = vtku.convertPointDataToNpArray(large_mesh, "scalars")

choice = subset_anneal(large_pts, large_el, num=small_size, runs=3000)
new_pts, new_el = subset_triangulate(large_pts, large_el, choice, holes=5)
dat = large_scar[choice]

if save2vtk:
    iou.cout("Save to vtk", print2console=verbose)
    nodes = vtk.vtkPoints()
    for ix in range(len(new_pts)):
        nodes.InsertPoint(ix, new_pts[ix, 0], new_pts[ix, 1], new_pts[ix, 2])

    elems = vtk.vtkCellArray()
    for ix in range(len(new_el)):
        el_id_list = vtk.vtkIdList()
        el_id_list.InsertNextId(new_el[ix, 0])
        el_id_list.InsertNextId(new_el[ix, 1])
        el_id_list.InsertNextId(new_el[ix, 2])
        elems.InsertNextCell(el_id_list)

    pd = vtk.vtkPolyData()
    pd.SetPoints(nodes)
    pd.SetPolys(elems)
    pd.GetPointData().SetScalars(vtknp.numpy_to_vtk(dat))
    p2c = vtk.vtkPointDataToCellData()
    p2c.SetInputData(pd)
    p2c.Update()
    vtku.writeVtk(p2c.GetOutput(), base_dir, "downsample_" + out_large_name)


# save to carp
if save2carp:
    iou.cout("Save to vtk", print2console=verbose)
    vtku.saveToCarpTxt(new_pts, new_el, iou.fullfile(base_dir, "downsample_" + out_large_name))
    np.savetxt(iou.fullfile(base_dir, "downsample_" + out_large_name + ".dat"), dat)
