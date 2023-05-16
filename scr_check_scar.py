import numpy as np
import vtk
import argparse

def vtk_version():
    return vtk.vtkVersion().GetVTKMajorVersion() + 0.1*vtk.vtkVersion().GetVTKMinorVersion()

def main(args) :
    default_output = args.output == ''

    base_dir = args.dir
    inname = args.input
    outname = args.input[1:-4] if default_output else args.output

    # remove extension from output name if exists
    outname += '.vtk' if not outname in '.vtk' else ''

    data = np.loadtxt(f'{base_dir}/{inname}', skiprows=3, delimiter=',')
    idx = data[:,0]
    centres = data[:, 1:4]
    normals =  data[:, -3:]
    normals_n= np.divide(normals.T,np.linalg.norm(normals, axis=1)).T

    min_step = -1
    max_step = 3

    min_step_positions = centres + min_step * normals_n
    max_step_positions = centres + max_step * normals_n

    polydata = vtk.vtkPolyData()
    points1 = vtk.vtkPoints()
    points2 = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for ix in range(len(min_step_positions)) :
        pt1 = min_step_positions[ix, :]
        pt2 = max_step_positions[ix, :]

        id1 = points1.InsertNextPoint(pt1)
        id2 = points2.InsertNextPoint(pt2)
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, id1)
        line.GetPointIds().SetId(1, id2)
        lines.InsertNextCell(line)

    polydata.SetPoints(points1)
    polydata.SetLines(lines)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(f'{base_dir}/{outname}')

    if vtk_version() >= 9.1 :
        writer.SetFileVersion(42)

    writer.Write()

if __name__ == '__main__' : 
    input_parser = argparse.ArgumentParser(description='Convert scar projection files to vtk')
    input_parser.add_argument("-d", "--dir", typ=str, required=True, help="Folder with data")
    input_parser.add_argument("-i", "--input", type=str, required=True, help="Input filename")
    input_parser.add_argument('-o', '--output', type=str, required=False, help='Output filename', default='')
    input_parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    input_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    args = input_parser.parse_args()
    
    main(args)