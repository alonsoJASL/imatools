import os
from imatools.common.ioutils import cout, fullfile
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
import pyvista as pv
import numpy as np
import random

import SimpleITK as sitk

def plot_dict(mydic, plotname, out_dir, oname, ylims=[]):
    """
    Plot dictionary.
    plot_params can be:
        [ymin ymax], xlabel, ylabel
    """
    fig, ax = plt.subplots()
    try:
        func = getattr(ax, plotname)
        func(mydic.values())
        ax.set_xticklabels(mydic.keys())

        if ylims:
            ax.set_ylim(ylims[0], ylims[1])

        fig.suptitle(oname)
        fig.savefig(fullfile(out_dir, '{}_{}.pdf'.format(oname, plotname)))
        return fig, ax

    except AttributeError:
        cout("Plot function {} does not exist".format(plotname), 'ERROR')
        sys.exit(-1)

def extract_scar_stats_from_file(filename: str) : 
    """
    Extracts scar stats from prodStats.txt file.
    """
    fname = os.path.normpath(filename)
    print(fname)
    scar_stats = {}
    with open(fname, 'r') as f : 
        lines = f.readlines()
        method = lines[0]
        bp_mean = lines[1]
        bp_std = lines[2]

        # format is 'V=value, SCORE=score'
        scar_stats = { 'value_score' : [] }
        for line in lines[3:] :
            line = line.strip()
            if line == '' : 
                continue
            values, score = line.split(',')
            values = values.split('=')[1]
            score = score.split('=')[1]
            scar_stats['value_score'].append((float(values), float(score)))
        
        scar_stats['method'] = method
        scar_stats['bp_mean'] = float(bp_mean)
        scar_stats['bp_std'] = float(bp_std)

    return scar_stats


def append_scar_to_pandas_dataframe(df: pd.DataFrame, scar_stats: dict, case_info: dict) :
    #case_id = '', roi_mode = '', roi_limits = '', thresh = '') : 
    """
    Append scar stats to pandas dataframe.
    """
    
    for tu in scar_stats['value_score'] :
        df = pd.concat([df, 
        pd.DataFrame({
            'case_id' : case_info['case_id'],
            'nav' : case_info['nav'], 
            'roi_mode' : case_info['roi_mode'],
            'roi_limits' : case_info['roi_limits'],
            'threshold_method' : case_info['thresh'],
            'bp_mean' : scar_stats['bp_mean'],
            'bp_std' : scar_stats['bp_std'],
            'values' : tu[0],
            'score' : tu[1]
            }, index=[0])], ignore_index=True)

    return df

def visualise_mesh(plt_msh,
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
                   dpi=300
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
    plotter.background_color = 'white'

    if colormap is not None:
        plotter.add_mesh(final_plt_msh,
                                scalars="ID",
                                cmap=colormap,
                                show_edges=False,
                                edge_color=None,
                                    opacity=opacity)
        plotter.remove_scalar_bar()
    else:
        plotter.add_mesh(final_plt_msh,
                                color=color,
                                show_edges=False,
                                edge_color=None,
                                    opacity=opacity)

        # Extract unique points used by the submesh
    submesh_cells = np.unique(final_plt_msh.cells, axis=0)
    submesh_points = final_plt_msh.points[submesh_cells,:]

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
    screenshot = plotter.screenshot(transparent_background=None, return_img=True, window_size=[fig_w, fig_h])

    plotter.close()

    # Save the screenshot to a PDF
    fig, ax = plt.subplots(figsize=(fig_w / 100 , fig_h / 100 ), dpi=dpi)
    ax.imshow(screenshot)
    ax.set_title(title, fontsize = title_fontsize, y=title_position)
    ax.axis('off')
    pdf_object.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def visualise_two_meshes(plt_msh_1,
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
                   dpi=300
                    ):

    print(f"Plotting {title}")

    

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'white'


    plotter.add_mesh(plt_msh_1,
                                color=color_1,
                                show_edges=False,
                                edge_color=None,
                                    opacity=opacity_mesh_1)
    
    plotter.add_mesh(plt_msh_2,
                                color=color_2,
                                show_edges=False,
                                edge_color=None,
                                    opacity=opacity_mesh_2)



    plotter.view_xz()

    submesh_cells = np.unique(plt_msh_1.cells, axis=0)
    submesh_points = plt_msh_1.points[submesh_cells,:]

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
    screenshot = plotter.screenshot(transparent_background=None, return_img=True, window_size=[fig_w, fig_h])

    plotter.close()

    # Save the screenshot to a PDF
    fig, ax = plt.subplots(figsize=(fig_w / 100 , fig_h / 100 ), dpi=dpi)
    ax.clear()
    ax.imshow(screenshot)
    ax.set_title(title, fontsize = title_fontsize, y=title_position)
    ax.axis('off')
    pdf_object.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def visualise_vtx(plt_msh,
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
                   dpi=300
                    ):

    print(f"Plotting {title}")

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'white'

    plotter.add_mesh(plt_msh,
                                color=color_mesh,
                                show_edges=False,
                                edge_color=None,
                                    opacity=opacity)
    
    points = plt_msh.points[vtx,:]

    plotter.add_points(points,
                       render_points_as_spheres=True,
                       color=color_vtx,
                       point_size = 50)

    plotter.view_xz()

    submesh_cells = np.unique(plt_msh.cells, axis=0)
    submesh_points = plt_msh.points[submesh_cells,:]

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
    screenshot = plotter.screenshot(transparent_background=None, return_img=True, window_size=[fig_w, fig_h])

    plotter.close()

    # Save the screenshot to a PDF
    fig, ax = plt.subplots(figsize=(fig_w / 100 , fig_h / 100 ), dpi=dpi)
    ax.imshow(screenshot)
    ax.set_title(title, fontsize = title_fontsize, y=title_position)
    ax.axis('off')
    pdf_object.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def visualise_fibres(plt_msh,
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
                     dpi=300):
    
    print(f"Plotting {title}")
    # Filter based on tags
    tags = plt_msh.cell_data["ID"]
    if type(tag) is int:
        tag = [tag]
    elif type(tag) is list:
        tag_filter = ([i in tag for i in tags])
    
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

    glyphs = final_submesh.glyph(orient='fibres', scale=False, factor=5000, geom=line.tube(radius=tube_radius, n_sides=20))
    
    # Plot the glyphs
    plotter = pv.Plotter(off_screen=True)
    plotter.enable_anti_aliasing('fxaa')
    plotter.background_color = 'white'

    colormap = plt.get_cmap(color)

    plotter.add_mesh(glyphs, color=colormap((tag[0] - 1) / 255))

    plotter.add_mesh(submesh, color='lightgray', opacity=0.1)

    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.view_xz()
    plotter.camera.roll += camera_roll_increment
    plotter.camera.azimuth += camera_azimuth_increment
    plotter.camera.elevation += camera_elevation_increment
    plotter.camera.zoom = zoom

    # Take screenshot
    screenshot = plotter.screenshot(transparent_background=None, return_img=True, window_size=[fig_w, fig_h])

    plotter.close()

    # Save the screenshot to a PDF
    fig, ax = plt.subplots(figsize=(fig_w / 100 , fig_h / 100 ), dpi=dpi)
    ax.imshow(screenshot)
    ax.set_title(title, fontsize=title_fontsize, y=title_position)
    ax.axis('off')
    pdf_object.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def visualise_pericardium(plt_msh,
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
                   pericardium_scale_path=None,
                   dpi=300
                    ):

    print(f"Plotting {title}")

    penalty_map_dat = np.genfromtxt(pericardium_scale_path,dtype=float)

    plt_msh["pericardium_scale"] = penalty_map_dat
    

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'white'

    plotter.add_mesh(plt_msh,
                               scalars="pericardium_scale",
                               cmap=colormap,
                               show_edges=False,
                               edge_color=None,
                                opacity=1)

    plotter.remove_scalar_bar()

    plotter.camera.focal_point = (0.0, 0.0, 0.0)

    plotter.camera_position = 'xz'
    plotter.camera.roll += camera_roll_increment
    plotter.camera.azimuth += camera_azimuth_increment
    plotter.camera.elevation += camera_elevation_increment
    plotter.camera.zoom = zoom
    print("Taking screenshot")

    # Take screenshot
    screenshot = plotter.screenshot(transparent_background=None, return_img=True, window_size=[fig_w, fig_h])

    plotter.close()
    print("Saving...")
    # Save the screenshot to a PDF
    fig, ax = plt.subplots(figsize=(fig_w / 100 , fig_h / 100 ), dpi=dpi)
    ax.imshow(screenshot)
    ax.set_title(title, fontsize = title_fontsize, y=title_position)
    ax.axis('off')
    pdf_object.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def visualise_3d_segmentation(seg: sitk.Image, save_as: str, name: str, show_fig=True) : 
    to_save = save_as != ''
    print('Loading segmentation')
    segmentation_np = sitk.GetArrayViewFromImage(seg)
    
    spacing = seg.GetSpacing()
    # dims = segmentation_np.shape
    # x = np.arange(dims[2]) * spacing[0]
    # y = np.arange(dims[1]) * spacing[1]
    # z = np.arange(dims[0]) * spacing[2]
    # x, y, z = np.meshgrid(x, y, z, indexing="ij")

    # # Create the pyvista structured grid with spacing accounted for
    # structured_grid = pv.StructuredGrid(x, y, z)

    # # Step 5: Add the segmentation values as cell scalars
    # structured_grid.cell_data['values'] = segmentation_np.flatten(order='F')

    unique_labels = np.unique(segmentation_np)
    # remove background
    unique_labels = unique_labels[1:]

    cmap = plt.cm.get_cmap('tab20', len(unique_labels)) # or 'coolwarm' or 'viridis'

    # print('Creating grid')
    # grid = pv.wrap(segmentation_np)
    # plotter.add_volume(grid, cmap='coolwarm', opacity='linear')

    print('Plotting...')
    plotter = pv.Plotter(title=name)
    for ix, label in enumerate(unique_labels) :
        if label == 0 :
            continue

        label_mask = segmentation_np == label
        label_grid = pv.wrap(label_mask.astype(np.uint8))
        label_grid = label_grid.scale(spacing)
        
        surface = label_grid.threshold(0.5).extract_surface()

        colour = cmap(ix)[:3]
        plotter.add_mesh(surface, color=colour, opacity=1.0)

    plotter.view_xz()
    if show_fig :
        plotter.show()

    if to_save :
        if name == '' :
            name = 'segmentation'
        plotter.screenshot(save_as)