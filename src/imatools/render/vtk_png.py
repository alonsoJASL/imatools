"""
VTK-to-PNG offscreen rendering.

Stateless: takes VTK file paths + render parameters, produces PNG image(s).
The low-level VTK reader/mapper/actor helpers (``create_vtk_reader``,
``create_vtk_mapper``, ``create_vtk_actor``) remain in
``imatools.common.vtktools`` (not part of this migration's scope). They are
imported lazily (inside the function bodies) because ``common.vtktools``
re-exports this module's public names via a bottom-of-file shim — a
module-level import here would create an import cycle.
"""

import os

import numpy as np
import vtk
from PIL import Image


def render_vtk_to_png(
    vtk_files, output_dir, window_size=(800, 600), input_type="ugrid", scalar_name="elemTag"
):
    from imatools.common.vtktools import create_vtk_actor, create_vtk_mapper, create_vtk_reader

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
    from imatools.common.vtktools import create_vtk_actor, create_vtk_mapper, create_vtk_reader

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
