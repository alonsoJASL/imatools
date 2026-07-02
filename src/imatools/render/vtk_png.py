"""
VTK-to-PNG offscreen rendering.

Stateless: takes VTK file paths + render parameters, produces PNG image(s).
The low-level VTK reader/mapper/actor helpers (``create_vtk_reader``,
``create_vtk_mapper``, ``create_vtk_actor``, ``center_vtk_data``) are defined
locally in this module.
"""

import os

import numpy as np
import vtk
from PIL import Image


def create_vtk_reader(input_type, filename, centered=False):
    """Return the appropriate VTK reader for the given input type and file."""
    if input_type == "ugrid":
        reader = vtk.vtkUnstructuredGridReader()
    else:  # 'polydata'
        reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    data = reader.GetOutput()
    if not data:
        raise ValueError(f"Failed to read VTK file: {filename}")
    if not isinstance(data, vtk.vtkDataSet):
        raise TypeError(f"Expected vtkDataSet, got {type(data)} for file: {filename}")

    # Center the data if requested
    if centered:
        data = center_vtk_data(data)

    return data


def center_vtk_data(data):
    """Translate the VTK dataset so that its geometric center is at (0,0,0)."""
    bounds = data.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    center = [
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ]

    transform = vtk.vtkTransform()
    transform.Translate(-center[0], -center[1], -center[2])

    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(data)
    transform_filter.Update()

    return transform_filter.GetOutput()


def create_vtk_mapper(input_type, data, scalar_name=None):
    """Return a VTK mapper configured for the given data and scalar coloring (if applicable)."""
    if input_type == "ugrid":
        mapper = vtk.vtkDataSetMapper()
    else:  # 'polydata'
        mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data)

    # Optionally handle scalar coloring
    if scalar_name:
        cell_data = data.GetCellData()
        point_data = data.GetPointData()
        if cell_data.HasArray(scalar_name):
            scalar_range = cell_data.GetArray(scalar_name).GetRange()
            mapper.SetScalarModeToUseCellData()
        elif point_data.HasArray(scalar_name):
            scalar_range = point_data.GetArray(scalar_name).GetRange()
            mapper.SetScalarModeToUsePointData()
        else:
            scalar_range = None  # Scalar field not found

        if scalar_range:
            lut = vtk.vtkLookupTable()
            lut.SetTableRange(scalar_range)
            lut.Build()
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(scalar_range)
            mapper.SelectColorArray(scalar_name)

    return mapper


def create_vtk_actor(mapper):
    """Return a VTK actor for the given mapper."""
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def render_vtk_to_png(
    vtk_files, output_dir, window_size=(800, 600), input_type="ugrid", scalar_name="elemTag"
):
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(window_size)

    for vtk_file in vtk_files:
        print(f"Rendering {vtk_file} to PNG...")

        # Read and center the data
        data = create_vtk_reader(input_type, vtk_file, centered=True)

        # # Debug: Print bounds before and after centering
        # original_bounds = data.GetBounds()
        # print(f"Centered mesh bounds: {original_bounds}")

        mapper = create_vtk_mapper(input_type, data, scalar_name)
        actor = create_vtk_actor(mapper)

        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)

        # Optional: Actor color for visibility
        # actor.GetProperty().SetColor(1, 0, 0)  # Red
        renderer.SetBackground(1, 1, 1)  # Light gray background

        render_window.AddRenderer(renderer)

        # **Reset the camera to fit this actor properly**
        renderer.ResetCamera()
        renderer.GetActiveCamera().Zoom(1.5)  # Optional: Zoom in slightly for better framing

        render_window.Render()

        # Capture the rendered image
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.ReadFrontBufferOff()  # Read from back buffer
        window_to_image_filter.Update()

        # Save to PNG
        writer = vtk.vtkPNGWriter()
        output_filename = f"{output_dir}/{os.path.basename(vtk_file).replace('.vtk', '.png')}"
        writer.SetFileName(output_filename)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

        print(f"Saved PNG: {output_filename}")

        # **Important: Clear the renderer before next file**
        render_window.RemoveRenderer(renderer)


def render_vtk_to_single_png(
    vtk_files,
    output_filename,
    grid_size=(1, 1),
    window_size=(800, 600),
    scalar_name="elemTag",
    input_type="ugrid",
    overlapping_margin=0.0,
    names=None,
):
    num_files = len(vtk_files)
    rows, cols = grid_size

    # sort files by name
    vtk_files = sorted(vtk_files)

    if rows * cols < num_files:
        raise ValueError("Grid size is too small to fit all VTK files.")

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(window_size[0] * cols, window_size[1] * rows)

    for i, vtk_file in enumerate(vtk_files):
        data = create_vtk_reader(input_type, vtk_file, centered=True)
        mapper = create_vtk_mapper(input_type, data, scalar_name)
        actor = create_vtk_actor(mapper)

        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(1, 1, 1)

        if names and i < len(names):
            vtk_name = os.path.basename(vtk_file)
            vtk_name = vtk_name.replace(".vtk", "").replace(".vtp", "")
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(vtk_name)
            text_actor.GetTextProperty().SetFontSize(24)
            text_actor.GetTextProperty().SetColor(0, 0, 0)
            text_actor.SetPosition(10, 10)
            renderer.AddActor2D(text_actor)

        margin = -overlapping_margin
        row, col = divmod(i, cols)
        viewport = [
            col / cols - margin,
            1 - (row + 1) / rows + margin,
            (col + 1) / cols + margin,
            1 - row / rows - margin,
        ]
        renderer.SetViewport(viewport)
        render_window.AddRenderer(renderer)
        print(f"Processing file {i+1}/{num_files}... [{row},{col}] - File: {vtk_file} ({vtk_name})")

    render_window.Render()
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    vtk_image = window_to_image_filter.GetOutput()
    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    image_array = vtk.util.numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, -1)
    image_array = np.flipud(image_array)

    image = Image.fromarray(image_array)
    image.save(output_filename)
