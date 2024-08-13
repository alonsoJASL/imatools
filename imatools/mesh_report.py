import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pyvista as pv
import random

from utils import rotate_mesh, read_pts, read_elem, read_lon, check_file, pts_elem_to_pyvista


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

def visualise_mesh_all_views(plt_msh,
                             fig_w,
                             fig_h,
                             colormap,
                             zoom,
                             pdf,
                             dpi,
                             title_fontsize,
                             title_position):
    ### Mesh from the anterior view

    visualise_mesh(plt_msh=plt_msh,
                fig_w=fig_w,
                fig_h=fig_h,
                colormap=colormap,
                zoom=zoom,
                pdf_object = pdf,
                title='Anterior view',
                dpi=dpi,
                title_fontsize=title_fontsize,
                title_position=title_position)


    ### Mesh from the posterior view
    visualise_mesh(plt_msh=plt_msh,
                fig_w=fig_w,
                fig_h=fig_h,
                colormap=colormap,
                zoom=zoom,
                pdf_object = pdf,
                camera_azimuth_increment=180,
                title='Posterior view',
                title_fontsize=title_fontsize,
                title_position=title_position,
                opacity=1.0,
                dpi=dpi)

            ### Mesh from the anterior view

    visualise_mesh(plt_msh=plt_msh,
                fig_w=fig_w,
                fig_h=fig_h,
                colormap=colormap,
                zoom=zoom,
                pdf_object = pdf,
                title='Anterior view - Translucent',
                title_fontsize=title_fontsize,
                title_position=title_position,
                opacity=0.7,
                dpi=dpi)


    ### Mesh from the posterior view - translucent
    visualise_mesh(plt_msh=plt_msh,
                fig_w=fig_w,
                fig_h=fig_h,
                colormap=colormap,
                zoom=zoom,
                pdf_object = pdf,
                camera_azimuth_increment=180,
                title='Posterior view - Translucent',
                title_fontsize=title_fontsize,
                title_position=title_position,
                opacity=0.7,
                dpi=dpi)

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

def visualise_fibres_all_views(plt_msh,
                                lon,
                                colormap,
                                pdf,
                                fig_w,
                                fig_h,
                                dpi,
                                zoom,
                                title_fontsize,
                                title_position):    

    visualise_fibres(plt_msh=plt_msh, 
                    tag = [1],
                    lon = lon, 
                    tube_radius = 0.03,
                    color = colormap,
                    pdf_object=pdf,
                    zoom=zoom,
                    fig_w = fig_w,
                    fig_h = fig_h,
                    title='Fibres (LV, anterior view)',
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    num_fibres = 10000,
                    dpi=dpi
                    )


    visualise_fibres(plt_msh=plt_msh, 
                    tag = [1],
                    lon = lon, 
                    tube_radius = 0.03,
                    color = colormap,
                    camera_elevation_increment=89,
                    pdf_object=pdf,
                    zoom=zoom,
                    fig_w = fig_w,
                    fig_h = fig_h,
                    title='Fibres (LV, basal view)',
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    num_fibres = 10000,
                    dpi=dpi
                    )
    visualise_fibres(plt_msh=plt_msh, 
                    tag = [1],
                    lon = lon, 
                    tube_radius = 0.03,
                    color = colormap,
                    camera_elevation_increment=-89,
                    pdf_object=pdf,
                    zoom=zoom,
                    fig_w = fig_w,
                    fig_h = fig_h,
                    title='Fibres (LV, apical view)',
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    num_fibres = 20000,
                    dpi=dpi
                    )
        
    visualise_fibres(plt_msh=plt_msh, 
                    tag = [2],
                    lon = lon, 
                    tube_radius = 0.03,
                    color = colormap,
                    camera_azimuth_increment=-90,
                    pdf_object=pdf,
                    zoom=zoom,
                    fig_w = fig_w,
                    fig_h = fig_h,
                    title='Fibres (RV, lateral view)',
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    num_fibres = 20000,
                    dpi=dpi
                    )

    visualise_fibres(plt_msh=plt_msh, 
                    tag = [3,26],
                    lon = lon, 
                    tube_radius = 0.03,
                    color = colormap,
                    pdf_object=pdf,
                    zoom=zoom,
                    fig_w = fig_w,
                    fig_h = fig_h,
                    title='Fibres (LA, anterior view)',
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    num_fibres=20000,
                    dpi=dpi
                    )


    visualise_fibres(plt_msh=plt_msh, 
                    tag = [3,26],
                    lon = lon, 
                    tube_radius = 0.03,
                    color = colormap,
                    camera_azimuth_increment=180,
                    pdf_object=pdf,
                    zoom=zoom,
                    fig_w = fig_w,
                    fig_h = fig_h,
                    title='Fibres (LA, posterior view)',
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    num_fibres=20000,
                    dpi=dpi
                    )


    visualise_fibres(plt_msh=plt_msh, 
                    tag = [3,26],
                    lon = lon, 
                    tube_radius = 0.03,
                    color = colormap,
                    camera_elevation_increment=89,
                    pdf_object=pdf,
                    zoom=zoom,
                    fig_w = fig_w,
                    fig_h = fig_h,
                    title='Fibres (LA, roof view)',
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    num_fibres=20000,
                    dpi=dpi
                    )

    visualise_fibres(plt_msh=plt_msh, 
                    tag = [4,26],
                    lon = lon, 
                    tube_radius = 0.03,
                    color = colormap,
                    camera_elevation_increment=89,
                    pdf_object=pdf,
                    zoom=zoom,
                    fig_w = fig_w,
                    fig_h = fig_h,
                    title='Fibres (RA, roof view)',
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    num_fibres=20000,
                    dpi=dpi
                    )

    visualise_fibres(plt_msh=plt_msh, 
                    tag = [4,26],
                    lon = lon, 
                    tube_radius = 0.03,
                    color = colormap,
                    camera_azimuth_increment=-90,
                    pdf_object=pdf,
                    zoom=zoom,
                    fig_w = fig_w,
                    fig_h = fig_h,
                    title='Fibres (RA, lateral view)',
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    num_fibres=20000,
                    dpi=dpi
                    )

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

def visualise_pericardium_all_views(meshname,
                                    plt_msh,
                                    pdf,
                                    fig_w,
                                    fig_h,
                                    colormap,
                                    dpi,
                                    zoom,
                                    title_fontsize,
                                    title_position):

    pericardium_scale_path = f"{'/'.join(meshname.split('/')[0:-1])}/pericardium_scale.dat"

    visualise_pericardium(plt_msh=plt_msh,
                    pdf_object=pdf,
                    fig_w=fig_w,
                    fig_h=fig_h,
                    colormap=colormap,
                    title="Pericardium scale (anterior view)",
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    pericardium_scale_path=pericardium_scale_path,
                    dpi=dpi,
                    zoom=zoom)

    visualise_pericardium(plt_msh=plt_msh,
                    pdf_object=pdf,
                    fig_w=fig_w,
                    fig_h=fig_h,
                    colormap=colormap,
                    camera_azimuth_increment=180,
                    title="Pericardium scale (posterior view)",
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    pericardium_scale_path=pericardium_scale_path,
                    dpi=dpi,
                    zoom=zoom)


    visualise_pericardium(plt_msh=plt_msh,
                    pdf_object=pdf,
                    fig_w=fig_w,
                    fig_h=fig_h,
                    colormap=colormap,
                    camera_azimuth_increment=0,
                    camera_elevation_increment=89,
                    title="Pericardium scale (top view)",
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    pericardium_scale_path=pericardium_scale_path,
                    dpi=dpi,
                    zoom=zoom)


    visualise_pericardium(plt_msh=plt_msh,
                    pdf_object=pdf,
                    fig_w=fig_w,
                    fig_h=fig_h,
                    colormap=colormap,
                    camera_azimuth_increment=0,
                    camera_elevation_increment=-89,
                    title="Pericardium scale (apical view)",
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    pericardium_scale_path=pericardium_scale_path,
                    dpi=dpi,
                    zoom=zoom)

def visualise_epicardium_all_views(meshname, 
                                   plt_msh,
                                   fig_w, 
                                   fig_h, 
                                   zoom, 
                                   pdf, 
                                   dpi, 
                                   title_fontsize, 
                                   title_position):
        
        elem_surf = read_elem(f"{'/'.join(meshname.split('/')[0:-1])}/epicardium_for_sim.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')


        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                pdf_object = pdf,
                title='Epicardium (anterior view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)

        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                camera_azimuth_increment=180,
                pdf_object = pdf,
                title='Epicardium (posterior view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)

def visualise_endocardia_all_views(meshname, 
                                   plt_msh, 
                                   fig_w, 
                                   fig_h, 
                                   zoom, 
                                   pdf, 
                                   dpi,
                                   title_fontsize,
                                   title_position):
        
        
        
        elem_surf = read_elem(f"{'/'.join(meshname.split('/')[0:-1])}/LV_endo.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')


        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                pdf_object = pdf,
                title='LV endocardium (anterior view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)

        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                camera_azimuth_increment=180,
                pdf_object = pdf,
                title='LV endocardium (posterior view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)
        
        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                camera_elevation_increment=89,
                pdf_object = pdf,
                title='LV endocardium (top view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)
        
                
        elem_surf = read_elem(f"{'/'.join(meshname.split('/')[0:-1])}/RV_endo.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')

        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                pdf_object = pdf,
                title='RV endocardium (septal view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi,
                camera_azimuth_increment=90)

        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                camera_azimuth_increment=-90,
                pdf_object = pdf,
                title='RV endocardium (lateral view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)
        
        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                camera_elevation_increment=89,
                camera_azimuth_increment=90,
                pdf_object = pdf,
                title='RV endocardium (basal view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)
        
        elem_surf = read_elem(f"{'/'.join(meshname.split('/')[0:-1])}/LA_endo.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')

        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                pdf_object = pdf,
                title='LA endocardium (anterior view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)

        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                camera_azimuth_increment=180,
                pdf_object = pdf,
                title='LA endocardium (posterior view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)
        
        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                camera_elevation_increment=-89,
                pdf_object = pdf,
                title='LA endocardium (basal view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)
        
        elem_surf = read_elem(f"{'/'.join(meshname.split('/')[0:-1])}/RA_endo.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')

        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                pdf_object = pdf,
                title='RA endocardium (septal view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi,
                camera_azimuth_increment=90
                )

        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                pdf_object = pdf,
                title='RA endocardium (lateral view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi,
                camera_azimuth_increment=-90
                )
        
        visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                pdf_object = pdf,
                title='RA endocardium (basal view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi,
                camera_azimuth_increment=-90,
                camera_elevation_increment=-89
                )

def visualise_veins_all_views(meshname, 
                              plt_msh, 
                              fig_w, 
                              fig_h, 
                              zoom, 
                              pdf, 
                              dpi,
                              title_fontsize,
                              title_position):
        
        
        
        elem_surf = read_elem(f"{'/'.join(meshname.split('/')[0:-1])}/RPVs.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')

        visualise_two_meshes(plt_msh_1 = plt_msh,
                             plt_msh_2 = plt_msh_surf,
                             fig_w=fig_w,
                             fig_h=fig_h,
                             color_1="lightgrey",
                             color_2="red",
                             zoom=zoom,
                             pdf_object = pdf,
                             title='Right pulmonary veins (location)',
                             title_fontsize=title_fontsize,
                             title_position=title_position,
                             dpi=dpi,
                             camera_elevation_increment=89,
                             opacity_mesh_1=0.3,
                             opacity_mesh_2=1)

                

        visualise_mesh(plt_msh=plt_msh_surf,
                       fig_w=fig_w,
                       fig_h=fig_h,
                       color="lightgrey",
                       zoom=zoom + 3,
                       pdf_object = pdf,
                       title='Right pulmonary veins (septal view)',
                       title_fontsize=title_fontsize,
                       title_position=title_position,
                       dpi=dpi,
                       camera_azimuth_increment=-90)

        visualise_mesh(plt_msh=plt_msh_surf,
                      fig_w=fig_w,
                      fig_h=fig_h,
                      color="lightgrey",
                      zoom=zoom+3,
                      pdf_object = pdf,
                      title='Right pulmonary veins (basal view)',
                      title_fontsize=title_fontsize,
                      title_position=title_position,
                      dpi=dpi,
                      camera_elevation_increment=-90)
        

        elem_surf = read_elem(f"{'/'.join(meshname.split('/')[0:-1])}/SVC.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')

        visualise_two_meshes(plt_msh_1 = plt_msh,
                             plt_msh_2 = plt_msh_surf,
                             fig_w=fig_w,
                             fig_h=fig_h,
                             color_1="lightgrey",
                             color_2="red",
                             zoom=zoom,
                             pdf_object = pdf,
                             title='Superior vena cava (location)',
                             title_fontsize=title_fontsize,
                             title_position=title_position,
                             dpi=dpi,
                             camera_elevation_increment=89,
                             opacity_mesh_1=0.3,
                             opacity_mesh_2=1)

                

        visualise_mesh(plt_msh=plt_msh_surf,
                      fig_w=fig_w,
                      fig_h=fig_h,
                      color="lightgrey",
                      zoom=zoom+3,
                      pdf_object = pdf,
                      title='Superior vena cava (top view)',
                      title_fontsize=title_fontsize,
                      title_position=title_position,
                      dpi=dpi,
                      camera_elevation_increment=89)

        visualise_mesh(plt_msh=plt_msh_surf,
                      fig_w=fig_w,
                      fig_h=fig_h,
                      color="lightgrey",
                      zoom=zoom+3,
                      pdf_object = pdf,
                      title='Superior vena cava (basal view)',
                      title_fontsize=title_fontsize,
                      title_position=title_position,
                      dpi=dpi,
                      camera_elevation_increment=-89)

def visualise_EAS_all_views(meshname, 
                            plt_msh, 
                            fig_w, 
                            fig_h, 
                            zoom, 
                            pdf, 
                            dpi,
                            title_fontsize,
                            title_position):
        
        
        
        vtx = np.genfromtxt(f"{'/'.join(meshname.split('/')[0:-1])}/SAN.vtx",skip_header=2, dtype=int)


        visualise_vtx(plt_msh = plt_msh,
                      vtx= vtx,
                      fig_w=fig_w,
                      fig_h=fig_h,
                      color_mesh="lightgrey",
                      color_vtx="red",
                      zoom=zoom,
                      pdf_object = pdf,
                      title='Sino-atrial node (top view)',
                      title_fontsize=title_fontsize,
                      title_position=title_position,
                      dpi=dpi,
                      camera_elevation_increment=89,
                      opacity=0.3)

        vtx = np.genfromtxt(f"{'/'.join(meshname.split('/')[0:-1])}/fascicles_lv.vtx",skip_header=2, dtype=int)


        visualise_vtx(plt_msh = plt_msh,
                      vtx= vtx,
                      fig_w=fig_w,
                      fig_h=fig_h,
                      color_mesh="lightgrey",
                      color_vtx="red",
                      zoom=zoom,
                      pdf_object = pdf,
                      title='LV fascicles (anterior view)',
                      title_fontsize=title_fontsize,
                      title_position=title_position,
                      dpi=dpi,
                      opacity=0.3)
        
        vtx = np.genfromtxt(f"{'/'.join(meshname.split('/')[0:-1])}/fascicles_rv.vtx",skip_header=2, dtype=int)


        visualise_vtx(plt_msh = plt_msh,
                      vtx= vtx,
                      fig_w=fig_w,
                      fig_h=fig_h,
                      color_mesh="lightgrey",
                      color_vtx="red",
                      zoom=zoom,
                      pdf_object = pdf,
                      title='RV fascicles (anterior view)',
                      title_fontsize=title_fontsize,
                      title_position=title_position,
                      dpi=dpi,
                      opacity=0.3)

def main(args):
    
    sims_folder = args.sims_folder
    report_name  = args.report_name

    fig_w = args.fig_w
    fig_h = args.fig_h

    colormap = args.colormap
    zoom = args.zoom
    dpi = args.dpi
    title_fontsize = args.title_fontsize
    title_position = args.title_position

    print_whole_mesh  = args.print_whole_mesh
    print_fibres      = args.print_fibres
    print_pericardium = args.print_pericardium
    print_epicardium  = args.print_epicardium
    print_endocardia  = args.print_endocardia
    print_veins       = args.print_veins
    print_EAS         = args.print_EAS
    print_all         = args.print_all

    if print_all:
        print_whole_mesh  = True
        print_fibres      = True
        print_pericardium = True
        print_epicardium  = True
        print_endocardia  = True
        print_veins       = True
        print_EAS         = True

    print_any = print_whole_mesh or print_fibres or print_pericardium or        print_epicardium or print_epicardium or print_endocardia or print_veins or print_EAS

    if not print_any:
        raise Exception("You need to choose what to print.")
    
    report_name_full = os.path.abspath(os.path.normpath(report_name))
    
    os.makedirs('/'.join(report_name_full.split('/')[0:-1]), exist_ok=True)

    meshname = f"{sims_folder}/myocardium_AV_FEC_BB_lvrv"

    check_file(f"{meshname}.pts")
    check_file(f"{meshname}.elem")

    if print_fibres:
        check_file(f"{meshname}.lon")
    if print_pericardium:
        check_file(f"{sims_folder}/pericardium_scale.dat")
    if print_epicardium:
        check_file(f"{sims_folder}/epicardium_for_sim.surf")
    if print_endocardia:
        check_file(f"{sims_folder}/LV_endo.surf")
        check_file(f"{sims_folder}/RV_endo.surf")
        check_file(f"{sims_folder}/LA_endo.surf")
        check_file(f"{sims_folder}/RA_endo.surf")
    if print_veins:
        check_file(f"{sims_folder}/RPVs.surf")
        check_file(f"{sims_folder}/SVC.surf")
    if print_EAS:
        check_file(f"{sims_folder}/SAN.vtx")
        check_file(f"{sims_folder}/fascicles_lv.vtx")
        check_file(f"{sims_folder}/fascicles_rv.vtx")
    

     ### We read the mesh once
      
    pts  = read_pts(meshname+'.pts')
    elem = read_elem(meshname+'.elem',el_type='Tt',tags=True)

    plt_msh = pts_elem_to_pyvista(pts=pts, elem=elem, add_tags=True)

    if print_fibres:
        lon_initial  = read_lon(meshname+".lon")
        plt_msh, lon = rotate_mesh(plt_msh = plt_msh, fibres = lon_initial[:,:3])
    else:
        plt_msh = rotate_mesh(plt_msh)
    
    with PdfPages(report_name) as pdf:
        
        if print_whole_mesh:
            visualise_mesh_all_views(plt_msh=plt_msh,
                                     fig_w=fig_w,
                                     fig_h=fig_h,
                                     colormap=colormap,
                                     zoom=zoom,
                                     pdf=pdf,
                                     dpi=dpi,
                                     title_fontsize=title_fontsize,
                                     title_position=title_position)
            
    ############# FIBRES #################

        if print_fibres:

            visualise_fibres_all_views(plt_msh=plt_msh,
                                       lon=lon,
                                       colormap=colormap,
                                       pdf=pdf,
                                       fig_w=fig_w,
                                       fig_h=fig_h,
                                       dpi=dpi,
                                       zoom=zoom,
                                       title_fontsize=title_fontsize,
                                       title_position=title_position)
               
        ###################### PERICARDIUM #########################

        if print_pericardium:

            visualise_pericardium_all_views(meshname=meshname,
                                            plt_msh=plt_msh,
                                            pdf=pdf,
                                            fig_w=fig_w,
                                            fig_h=fig_h,
                                            colormap=colormap,
                                            dpi=dpi,
                                            zoom=zoom,
                                            title_fontsize=title_fontsize,
                                            title_position=title_position)    
            
    ######################### SURFACES ###############################

    ######### EPICARDIUM ###########

        if print_epicardium:
            visualise_epicardium_all_views(meshname=meshname,
                                           plt_msh=plt_msh,
                                           pdf=pdf,
                                           fig_w=fig_w,
                                           fig_h=fig_h,
                                           zoom=zoom,
                                           dpi=dpi,
                                           title_fontsize=title_fontsize,
                                           title_position=title_position)
    
    ######### ENDOCARDIA ######### 
        if print_endocardia:
            visualise_endocardia_all_views(meshname=meshname,
                                           plt_msh=plt_msh,
                                           pdf=pdf,
                                           fig_w=fig_w,
                                           fig_h=fig_h,
                                           zoom=zoom,
                                           dpi=dpi,
                                           title_fontsize=title_fontsize,
                                           title_position=title_position)

    ######### VEINS ##########
        if print_veins:
            visualise_veins_all_views(meshname=meshname,
                                      plt_msh=plt_msh,
                                      pdf=pdf,
                                      fig_w=fig_w,
                                      fig_h=fig_h,
                                      zoom=zoom,
                                      dpi=dpi,
                                      title_fontsize=title_fontsize,
                                      title_position=title_position)
            
        if print_EAS:
            visualise_EAS_all_views(meshname=meshname,
                                    plt_msh=plt_msh,
                                    pdf=pdf,
                                    fig_w=fig_w,
                                    fig_h=fig_h,
                                    zoom=zoom,
                                    dpi=dpi,
                                    title_fontsize=title_fontsize,
                                    title_position=title_position)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
	   description='Creates a report with images for mesh quality assessment.'
    )
    
    parser.add_argument(
	   '--sims_folder',
	   type=str,
	   help='Path to the folder containing simulation-ready files.'
    )
    parser.add_argument(
	   '--report_name',
	   type=str,
	   default='output.pdf',
	   help='Output file name.'
    )
    parser.add_argument(
	   '--fig_w',
	   type=int,
	   default=2480,
	   help='Page width in pixels.'
    )
    parser.add_argument(
	   '--fig_h',
	   type=int,
	   default=3508,
	   help='Page height in pixels.'
    )
    parser.add_argument(
	   '--colormap',
	   type=str,
	   default='RdBu',
	   help='Matplotlib colormap for tags.'
    )
    parser.add_argument(
	   '--zoom',
	   type=float,
	   default=1,
	   help='Zoom magnitude.'
    )
    parser.add_argument(
	   '--dpi',
	   type=float,
	   default=100,
	   help='Dots per inch (resolution).'
    )
    parser.add_argument(
	   '--title_fontsize',
	   type=float,
	   default=44,
	   help='Fontsize of the title of each page.'
    )
    parser.add_argument(
	   '--title_position',
	   type=float,
	   default=0.9,
	   help='Title position value. 1 is at the top of the page, the lower the value, the lower the position of the title'
    )
    parser.add_argument(
	   '--print_whole_mesh',
	   action=argparse.BooleanOptionalAction,
	   default=False,
	   help='Include anterior and posterior mesh views (opaque and translucent).'
    )
    parser.add_argument(
	   '--print_fibres',
	   action=argparse.BooleanOptionalAction,
	   default=False,
	   help='Include mesh with fibres, separated by chamber, from different views.'
    )
    parser.add_argument(
	   '--print_pericardium',
	   action=argparse.BooleanOptionalAction,
	   default=False,
	   help='Include mesh with pericardium penalty map from different views.'
    )
    parser.add_argument(
	   '--print_epicardium',
	   action=argparse.BooleanOptionalAction,
	   default=False,
	   help='Include epicardium surface views.'
    )
    parser.add_argument(
	   '--print_endocardia',
	   action=argparse.BooleanOptionalAction,
	   default=False,
	   help='Include endocardia surface views.'
    )
    parser.add_argument(
	   '--print_veins',
	   action=argparse.BooleanOptionalAction,
	   default=False,
	   help='Include right pulmonary veins and superior vena cava surface views.'
    )
    parser.add_argument(
	   '--print_EAS',
	   action=argparse.BooleanOptionalAction,
	   default=False,
	   help='Include early activation sites (sino-atrial node and ventricular fascicles) views.'
    )
    parser.add_argument(
	   '--print_all',
	   action=argparse.BooleanOptionalAction,
	   default=False,
        help='Include all possible images in the report.'
    )

    args = parser.parse_args()
    main(args)