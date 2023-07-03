import common.vtktools as vtku 
import numpy as np
import vtk
import os 
from vtk.util import numpy_support
import argparse

def flip_xy(polydata) :
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()

    for i in range(num_points):
        original_coords = points.GetPoint(i)
        modified_coords = [-original_coords[0], -original_coords[1], original_coords[2]]
        points.SetPoint(i, modified_coords)

    polydata.Modified()

def main(args) : 
    p2f = args.dir
    fname = os.path.join(p2f, args.input)
    output = args.output if args.output != '' else 'segmentation'

    mesh = vtku.readVtk(fname)
    segvtk = vtk.vtkPolyData()
    segvtk.DeepCopy(mesh)

    flip_xy(segvtk)
    vtku.writeVtk(segvtk, os.path.dirname(fname), output)

    print(f'IN:{fname} OUT:{os.path.join(os.path.dirname(fname), output)}.vtk')

if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(description='Flip XY coordinates of a polydata')
    parser.add_argument('--dir', help='input file', type=str, required=True)
    parser.add_argument('--input', help='input file', type=str, required=True)
    parser.add_argument('--output', help='output file', type=str, default='')
    args = parser.parse_args()

    main(args)
