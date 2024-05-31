import os
import argparse

from common import itktools as itku
from common import vtktools as vtku 

def main(args):
    # Load the vtk file
    vtk_file = args.input
    vtku.load_vtk(vtk_file)

    # Export the vtk file
    vtku.export_vtk_as(vtk_file, f'{args.output}.{args.mode}', args.mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export vtk file as another format')
    parser.add_argument('mode', choices=['ply', 'stl', 'obj', 'vtp'],  help='Export as format')
    parser.add_argument('--input', help='VTK file to load')
    parser.add_argument('--output', help='Output file name')
    args = parser.parse_args()
    main(args)