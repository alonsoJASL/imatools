import numpy as np
import vtk
import argparse

import imatools.common.vtktools as vtku 
import imatools.common.ioutils as iou 

def vtk_version():
    return vtk.vtkVersion().GetVTKMajorVersion() + 0.1*vtk.vtkVersion().GetVTKMinorVersion()

def main(args) :
    default_output = args.output == ''

    base_dir = args.dir
    inname = args.input
    outname = args.input[0:-4] if default_output else args.output

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

    if args.verbose :
        print(f'min step: {min_step}')
        print(min_step_positions)
        print(f'max step: {max_step}')
        print(max_step_positions)
        print("normals")
        print(normals_n)
    
    polydata = vtk.vtkPolyData()
    lines = vtk.vtkCellArray()
    pts = vtk.vtkPoints()

    num_points1 = len(min_step_positions)

    vf = vtk.vtkDoubleArray()
    vf.SetName("scar_corridor")
    vf.SetNumberOfComponents(3)
    vf.SetNumberOfTuples(num_points1)
    
    for i in range(num_points1):
        pt1 = min_step_positions[i]
        pt2 = max_step_positions[i]

        vf.SetTuple3(i, pt2[0] - pt1[0], pt2[1] - pt1[1], pt2[2] - pt1[2])

    polydata.SetPoints(pts)
    polydata.GetPointData().SetVectors(vf)

    if args.verbose :
        # Create a vtkArrowSource for generating the arrows
        arrow_source = vtk.vtkArrowSource()

        # Create a vtkGlyph3D to generate the arrows along the lines
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(arrow_source.GetOutputPort())
        glyph.SetInputData(polydata)
        glyph.Update()

        # Create a vtkPolyDataMapper to map the glyph's output data for visualization
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        # Create a vtkActor and set the mapper for visualization
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create a vtkRenderer and add the actor to it
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)

        # Create a vtkRenderWindow and set the renderer for visualization
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)

        # Create a vtkRenderWindowInteractor for interaction
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        # Set up the interactor and start the rendering loop
        interactor.Initialize()
        render_window.Render()
        interactor.Start()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(f'{base_dir}/{outname}')

    if vtk_version() >= 9.1 :
        writer.SetFileVersion(42)

    writer.Write()

if __name__ == '__main__' : 
    input_parser = argparse.ArgumentParser(description='Convert scar projection files to vtk')
    input_parser.add_argument("-d", "--dir", type=str, required=True, help="Folder with data")
    input_parser.add_argument("-i", "--input", type=str, required=True, help="Input filename")
    input_parser.add_argument('-o', '--output', type=str, required=False, help='Output filename', default='')
    input_parser.add_argument("-n", "--number-")
    input_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    args = input_parser.parse_args()
    
    main(args)