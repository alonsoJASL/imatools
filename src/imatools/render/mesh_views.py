"""
Pyvista-based mesh visualisation.

Stateless: takes mesh/point data + render parameters + an already-open PDF
(or figure) handle, produces a rendered view. No file-path resolution or
hardcoded-path I/O lives here.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk


def pts_elem_to_pyvista(pts, elem, add_tags=False, el_type="Tt"):

    tmp_elem = elem

    if el_type == "Tt":
        final_elem = tmp_elem[:, :4]
        tets = np.column_stack(
            (np.ones((final_elem.shape[0],), dtype=int) * 4, final_elem)
        ).flatten()
        cell_type = np.ones((final_elem.shape[0],), dtype=int) * vtk.VTK_TETRA
    elif el_type == "Tr":
        final_elem = tmp_elem[:, :3]
        tets = np.column_stack(
            (np.ones((final_elem.shape[0],), dtype=int) * 3, final_elem)
        ).flatten()
        cell_type = np.ones((final_elem.shape[0],), dtype=int) * vtk.VTK_TRIANGLE

    plt_msh = pv.UnstructuredGrid(tets, cell_type, pts)
    if add_tags:
        tags = tmp_elem[:, -1]
        plt_msh.cell_data["ID"] = tags

    return plt_msh


def visualise_mesh(
    plt_msh,
    pdf_object,
    fig_w=500,
    fig_h=500,
    colormap=None,
    color="lightgrey",
    zoom=2,
    camera_roll_increment=0,
    camera_azimuth_increment=0,
    camera_elevation_increment=0,
    title=None,
    title_fontsize=44,
    title_position=0.9,
    opacity=1.0,
    dpi=300,
):

    print(f"Plotting {title}")

    ###### TRANSLATING THE MESH TO CENTRE IT DIDN'T WORK ###
    # if translate_mesh:

    #     cog = np.mean(plt_msh.points[np.unique(plt_msh.cells),:],axis=0)

    #     print(cog)

    #     pts_transformed = plt_msh.points-cog

    #     final_plt_msh = plt_msh.copy()

    #     final_plt_msh.points = pts_transformed
    # else:
    #     final_plt_msh = plt_msh
    #

    final_plt_msh = plt_msh

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = "white"

    if colormap is not None:
        plotter.add_mesh(
            final_plt_msh,
            scalars="ID",
            cmap=colormap,
            show_edges=False,
            edge_color=None,
            opacity=opacity,
        )
        plotter.remove_scalar_bar()
    else:
        plotter.add_mesh(
            final_plt_msh, color=color, show_edges=False, edge_color=None, opacity=opacity
        )

        # Extract unique points used by the submesh
    submesh_cells = np.unique(final_plt_msh.cells, axis=0)
    submesh_points = final_plt_msh.points[submesh_cells, :]

    # Calculate the center of the submesh based on these points
    center = np.mean(submesh_points, axis=0)

    plotter.view_xz()

    # Update the camera focal point to the center
    plotter.camera.focal_point = center

    # plotter.camera.focal_point = (0.0, 0.0, 0.0)

    plotter.camera.roll += camera_roll_increment
    plotter.camera.azimuth += camera_azimuth_increment
    plotter.camera.elevation += camera_elevation_increment

    plotter.camera.zoom(zoom)
    print("Camera setup")

    # Take screenshot
    screenshot = plotter.screenshot(
        transparent_background=None, return_img=True, window_size=[fig_w, fig_h]
    )

    plotter.close()

    # Save the screenshot to a PDF
    fig, ax = plt.subplots(figsize=(fig_w / 100, fig_h / 100), dpi=dpi)
    ax.imshow(screenshot)
    ax.set_title(title, fontsize=title_fontsize, y=title_position)
    ax.axis("off")
    pdf_object.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def visualise_two_meshes(
    plt_msh_1,
    plt_msh_2,
    pdf_object,
    fig_w=500,
    fig_h=500,
    color_1="lightgrey",
    color_2="red",
    zoom=2,
    camera_roll_increment=0,
    camera_azimuth_increment=0,
    camera_elevation_increment=0,
    title=None,
    title_fontsize=44,
    title_position=0.9,
    opacity_mesh_1=0.3,
    opacity_mesh_2=1.0,
    dpi=300,
):

    print(f"Plotting {title}")

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = "white"

    plotter.add_mesh(
        plt_msh_1, color=color_1, show_edges=False, edge_color=None, opacity=opacity_mesh_1
    )

    plotter.add_mesh(
        plt_msh_2, color=color_2, show_edges=False, edge_color=None, opacity=opacity_mesh_2
    )

    plotter.view_xz()

    submesh_cells = np.unique(plt_msh_1.cells, axis=0)
    submesh_points = plt_msh_1.points[submesh_cells, :]

    # Calculate the center of the submesh based on these points
    center = np.mean(submesh_points, axis=0)

    # Update the camera focal point to the center
    plotter.camera.focal_point = center

    plotter.camera.roll += camera_roll_increment
    plotter.camera.azimuth += camera_azimuth_increment
    plotter.camera.elevation += camera_elevation_increment

    plotter.camera.zoom(zoom)
    print("Camera setup")

    # Take screenshot
    screenshot = plotter.screenshot(
        transparent_background=None, return_img=True, window_size=[fig_w, fig_h]
    )

    plotter.close()

    # Save the screenshot to a PDF
    fig, ax = plt.subplots(figsize=(fig_w / 100, fig_h / 100), dpi=dpi)
    ax.clear()
    ax.imshow(screenshot)
    ax.set_title(title, fontsize=title_fontsize, y=title_position)
    ax.axis("off")
    pdf_object.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def visualise_vtx(
    plt_msh,
    vtx,
    pdf_object,
    fig_w=500,
    fig_h=500,
    color_mesh="lightgrey",
    color_vtx="red",
    zoom=2,
    camera_roll_increment=0,
    camera_azimuth_increment=0,
    camera_elevation_increment=0,
    title=None,
    title_fontsize=44,
    title_position=0.9,
    opacity=0.3,
    dpi=300,
):

    print(f"Plotting {title}")

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = "white"

    plotter.add_mesh(plt_msh, color=color_mesh, show_edges=False, edge_color=None, opacity=opacity)

    points = plt_msh.points[vtx, :]

    plotter.add_points(points, render_points_as_spheres=True, color=color_vtx, point_size=50)

    plotter.view_xz()

    submesh_cells = np.unique(plt_msh.cells, axis=0)
    submesh_points = plt_msh.points[submesh_cells, :]

    # Calculate the center of the submesh based on these points
    center = np.mean(submesh_points, axis=0)

    # Update the camera focal point to the center
    plotter.camera.focal_point = center

    plotter.camera.roll += camera_roll_increment
    plotter.camera.azimuth += camera_azimuth_increment
    plotter.camera.elevation += camera_elevation_increment

    plotter.camera.zoom(zoom)
    print("Camera setup")

    # Take screenshot
    screenshot = plotter.screenshot(
        transparent_background=None, return_img=True, window_size=[fig_w, fig_h]
    )

    plotter.close()

    # Save the screenshot to a PDF
    fig, ax = plt.subplots(figsize=(fig_w / 100, fig_h / 100), dpi=dpi)
    ax.imshow(screenshot)
    ax.set_title(title, fontsize=title_fontsize, y=title_position)
    ax.axis("off")
    pdf_object.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def visualise_fibres(
    plt_msh,
    lon,
    tag=1,
    tube_radius=0.08,
    color="red",
    camera_roll_increment=0,
    camera_azimuth_increment=0,
    camera_elevation_increment=0,
    title=None,
    title_fontsize=44,
    title_position=0.9,
    pdf_object=None,
    zoom=2,
    fig_w=500,
    fig_h=500,
    num_fibres=4000,
    dpi=300,
):

    print(f"Plotting {title}")
    # Filter based on tags
    tags = plt_msh.cell_data["ID"]
    if type(tag) is int:
        tag = [tag]
    elif type(tag) is list:
        tag_filter = [i in tag for i in tags]

    # Extract the submesh with the specified tags
    submesh = plt_msh.extract_cells(tag_filter)

    # Filter fibres based on the submesh
    fibres = lon[tag_filter, :3]
    nelem = submesh.n_cells

    # Subsample fibres
    nelem_nofibres = nelem - num_fibres

    print(f"Plotting {nelem - nelem_nofibres} fibres")
    exclude = random.sample(range(0, nelem), nelem_nofibres)
    exclude_filter = np.ones(nelem, dtype=bool)
    exclude_filter[exclude] = 0

    final_submesh = submesh.extract_cells(exclude_filter)

    # Add the filtered fibres to the submesh
    final_submesh["fibres"] = fibres[exclude_filter, :]

    # Create glyphs for fibres
    line = pv.Line(resolution=300)

    glyphs = final_submesh.glyph(
        orient="fibres", scale=False, factor=5000, geom=line.tube(radius=tube_radius, n_sides=20)
    )

    # Plot the glyphs
    plotter = pv.Plotter(off_screen=True)
    plotter.enable_anti_aliasing("fxaa")
    plotter.background_color = "white"

    colormap = plt.get_cmap(color)

    plotter.add_mesh(glyphs, color=colormap((tag[0] - 1) / 255))

    plotter.add_mesh(submesh, color="lightgray", opacity=0.1)

    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.view_xz()
    plotter.camera.roll += camera_roll_increment
    plotter.camera.azimuth += camera_azimuth_increment
    plotter.camera.elevation += camera_elevation_increment
    plotter.camera.zoom = zoom

    # Take screenshot
    screenshot = plotter.screenshot(
        transparent_background=None, return_img=True, window_size=[fig_w, fig_h]
    )

    plotter.close()

    # Save the screenshot to a PDF
    fig, ax = plt.subplots(figsize=(fig_w / 100, fig_h / 100), dpi=dpi)
    ax.imshow(screenshot)
    ax.set_title(title, fontsize=title_fontsize, y=title_position)
    ax.axis("off")
    pdf_object.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def visualise_pericardium(
    plt_msh,
    pdf_object,
    fig_w=500,
    fig_h=500,
    colormap=None,
    zoom=2,
    camera_roll_increment=0,
    camera_azimuth_increment=0,
    camera_elevation_increment=0,
    title=None,
    title_fontsize=44,
    title_position=0.9,
    pericardium_scale=None,
    dpi=300,
):

    print(f"Plotting {title}")

    plt_msh["pericardium_scale"] = pericardium_scale

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = "white"

    plotter.add_mesh(
        plt_msh,
        scalars="pericardium_scale",
        cmap=colormap,
        show_edges=False,
        edge_color=None,
        opacity=1,
    )

    plotter.remove_scalar_bar()

    plotter.camera.focal_point = (0.0, 0.0, 0.0)

    plotter.camera_position = "xz"
    plotter.camera.roll += camera_roll_increment
    plotter.camera.azimuth += camera_azimuth_increment
    plotter.camera.elevation += camera_elevation_increment
    plotter.camera.zoom = zoom
    print("Taking screenshot")

    # Take screenshot
    screenshot = plotter.screenshot(
        transparent_background=None, return_img=True, window_size=[fig_w, fig_h]
    )

    plotter.close()
    print("Saving...")
    # Save the screenshot to a PDF
    fig, ax = plt.subplots(figsize=(fig_w / 100, fig_h / 100), dpi=dpi)
    ax.imshow(screenshot)
    ax.set_title(title, fontsize=title_fontsize, y=title_position)
    ax.axis("off")
    pdf_object.savefig(fig, bbox_inches="tight")
    plt.close(fig)
