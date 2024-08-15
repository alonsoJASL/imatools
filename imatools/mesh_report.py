import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pyvista as pv

from imatools.common.utils import rotate_mesh, pts_elem_to_pyvista
import imatools.common.ioutils as iou 
import imatools.common.plotutils as pu

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
    options = [
        ('Anterior view', 0, 1.0), 
        ('Posterior view', 180, 1.0),
        ('Anterior view - Translucent', 0, 0.7),
        ('Posterior view - Translucent', 180, 0.7)
    ]

    for title, az_incr, op in options:
        pu.visualise_mesh(plt_msh=plt_msh,
                    fig_w=fig_w,
                    fig_h=fig_h,
                    colormap=colormap,
                    zoom=zoom,
                    pdf_object = pdf,
                    camera_azimuth_increment=az_incr,
                    title=title,
                    title_fontsize=title_fontsize,
                    title_position=title_position,
                    opacity=op,
                    dpi=dpi)

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
    options = [
        # title, tag list, [roll, azimuth, elevation], num_fibres
        ('LV fibres (anterior view)', [1], [0, 0, 0], 10000),
        ('LV fibres (basal view)', [1], [0, 0, 89], 10000),
        ('LV fibres (apical view)', [1], [0, 0, -89], 20000),
        ('RV fibres (lateral view)', [2], [0, -90, 0], 20000),
        ('LA fibres (anterior view)', [3,26], [0, 0, 0], 20000),
        ('LA fibres (posterior view)', [3,26], [0, 180, 0], 20000),
        ('LA fibres (roof view)', [3,26], [0, 0, 89], 20000),
        ('RA fibres (roof view)', [4,26], [0, 0, 89], 20000),
        ('RA fibres (lateral view)', [4,26], [0, -90, 0], 20000)
    ]

    for mytitle, tag, camera, num_fibres in options:
        pu.visualise_fibres(plt_msh=plt_msh,
                         lon=lon,
                         tag=tag,
                         color=colormap,
                         camera_roll_increment=camera[0],
                         camera_azimuth_increment=camera[1],
                         camera_elevation_increment=camera[2],
                         title=mytitle,
                         title_fontsize=title_fontsize,
                         title_position=title_position,
                         pdf_object=pdf,
                         zoom=zoom,
                         fig_w=fig_w,
                         fig_h=fig_h,
                         num_fibres=num_fibres,
                         dpi=dpi)

def visualise_pericardium_all_views(sims_folder,
                                    plt_msh,
                                    pdf,
                                    fig_w,
                                    fig_h,
                                    colormap,
                                    dpi,
                                    zoom,
                                    title_fontsize,
                                    title_position):
    
    pericardium_scale_path = f"{sims_folder}/pericardium_scale.dat"

    options = [
        # title, azimuth, elevation
        ('Pericardium scale (anterior view)', 0, 0),
        ('Pericardium scale (posterior view)', 180, 0),
        ('Pericardium scale (top view)', 0, 89),
        ('Pericardium scale (apical view)', 0, -89)
    ]

    for mytitle, az_incr, el_incr in options:
        pu.visualise_pericardium(plt_msh=plt_msh,
                             pdf_object=pdf,
                             fig_w=fig_w,
                             fig_h=fig_h,
                             colormap=colormap,
                             camera_azimuth_increment=az_incr,
                             camera_elevation_increment=el_incr,
                             title=mytitle,
                             title_fontsize=title_fontsize,
                             title_position=title_position,
                             pericardium_scale_path=pericardium_scale_path,
                             dpi=dpi,
                             zoom=zoom)

def visualise_epicardium_all_views(sims_folder, 
                                   plt_msh,
                                   fig_w, 
                                   fig_h, 
                                   zoom, 
                                   pdf, 
                                   dpi, 
                                   title_fontsize, 
                                   title_position):
        
        elem_surf = iou.read_elem(f"{sims_folder}/epicardium_for_sim.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')

        pu.visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                pdf_object = pdf,
                title='Epicardium (anterior view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)

        pu.visualise_mesh(plt_msh=plt_msh_surf,
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

def visualise_endocardia_all_views(sims_folder, 
                                   plt_msh, 
                                   fig_w, 
                                   fig_h, 
                                   zoom, 
                                   pdf, 
                                   dpi,
                                   title_fontsize,
                                   title_position):
        
        
        
        elem_surf = iou.read_elem(f"{sims_folder}/LV_endo.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')


        pu.visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                pdf_object = pdf,
                title='LV endocardium (anterior view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)

        pu.visualise_mesh(plt_msh=plt_msh_surf,
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
        
        pu.visualise_mesh(plt_msh=plt_msh_surf,
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
        
                
        elem_surf = iou.read_elem(f"{sims_folder}/RV_endo.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')

        pu.visualise_mesh(plt_msh=plt_msh_surf,
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

        pu.visualise_mesh(plt_msh=plt_msh_surf,
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
        
        pu.visualise_mesh(plt_msh=plt_msh_surf,
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
        
        elem_surf = iou.read_elem(f"{sims_folder}/LA_endo.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')

        pu.visualise_mesh(plt_msh=plt_msh_surf,
                fig_w=fig_w,
                fig_h=fig_h,
                color="lightgrey",
                zoom=zoom,
                pdf_object = pdf,
                title='LA endocardium (anterior view)',
                title_fontsize=title_fontsize,
                title_position=title_position,
                dpi=dpi)

        pu.visualise_mesh(plt_msh=plt_msh_surf,
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
        
        pu.visualise_mesh(plt_msh=plt_msh_surf,
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
        
        elem_surf = iou.read_elem(f"{sims_folder}/RA_endo.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')

        pu.visualise_mesh(plt_msh=plt_msh_surf,
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

        pu.visualise_mesh(plt_msh=plt_msh_surf,
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
        
        pu.visualise_mesh(plt_msh=plt_msh_surf,
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

def visualise_veins_all_views(sims_folder, 
                              plt_msh, 
                              fig_w, 
                              fig_h, 
                              zoom, 
                              pdf, 
                              dpi,
                              title_fontsize,
                              title_position):
        
        
        
        elem_surf = iou.read_elem(f"{sims_folder}/RPVs.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')

        pu.visualise_two_meshes(plt_msh_1 = plt_msh,
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

        pu.visualise_mesh(plt_msh=plt_msh_surf,
                       fig_w=fig_w,
                       fig_h=fig_h,
                       color="lightgrey",
                       zoom=zoom,
                       pdf_object = pdf,
                       title='Right pulmonary veins (septal view)',
                       title_fontsize=title_fontsize,
                       title_position=title_position,
                       dpi=dpi,
                       camera_azimuth_increment=-90)

        pu.visualise_mesh(plt_msh=plt_msh_surf,
                      fig_w=fig_w,
                      fig_h=fig_h,
                      color="lightgrey",
                      zoom=zoom,
                      pdf_object = pdf,
                      title='Right pulmonary veins (basal view)',
                      title_fontsize=title_fontsize,
                      title_position=title_position,
                      dpi=dpi,
                      camera_elevation_increment=-90)
        

        elem_surf = iou.read_elem(f"{sims_folder}/SVC.surf",el_type='Tr',tags=False)

        plt_msh_surf = pts_elem_to_pyvista(pts=plt_msh.points, elem=elem_surf, add_tags=False, el_type='Tr')

        pu.visualise_two_meshes(plt_msh_1 = plt_msh,
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

                

        pu.visualise_mesh(plt_msh=plt_msh_surf,
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

        pu.visualise_mesh(plt_msh=plt_msh_surf,
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

def visualise_EAS_all_views(sims_folder, 
                            plt_msh, 
                            fig_w, 
                            fig_h, 
                            zoom, 
                            pdf, 
                            dpi,
                            title_fontsize,
                            title_position):
        
        
        
        vtx = np.genfromtxt(f"{sims_folder}/SAN.vtx",skip_header=2, dtype=int)


        pu.visualise_vtx(plt_msh = plt_msh,
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

        vtx = np.genfromtxt(f"{sims_folder}/fascicles_lv.vtx",skip_header=2, dtype=int)


        pu.visualise_vtx(plt_msh = plt_msh,
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
        
        vtx = np.genfromtxt(f"{sims_folder}/fascicles_rv.vtx",skip_header=2, dtype=int)


        pu.visualise_vtx(plt_msh = plt_msh,
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
    report_name_path = os.path.dirname(report_name_full)
    
    os.makedirs(report_name_path, exist_ok=True)

    meshname = f"{sims_folder}/myocardium_AV_FEC_BB_lvrv"

    iou.check_file(f"{meshname}.pts")
    iou.check_file(f"{meshname}.elem")

    if print_fibres:
        iou.check_file(f"{meshname}.lon")
    if print_pericardium:
        iou.check_file(f"{sims_folder}/pericardium_scale.dat")
    if print_epicardium:
        iou.check_file(f"{sims_folder}/epicardium_for_sim.surf")
    if print_endocardia:
        iou.check_file(f"{sims_folder}/LV_endo.surf")
        iou.check_file(f"{sims_folder}/RV_endo.surf")
        iou.check_file(f"{sims_folder}/LA_endo.surf")
        iou.check_file(f"{sims_folder}/RA_endo.surf")
    if print_veins:
        iou.check_file(f"{sims_folder}/RPVs.surf")
        iou.check_file(f"{sims_folder}/SVC.surf")
    if print_EAS:
        iou.check_file(f"{sims_folder}/SAN.vtx")
        iou.check_file(f"{sims_folder}/fascicles_lv.vtx")
        iou.check_file(f"{sims_folder}/fascicles_rv.vtx")
    

     ### We read the mesh once
      
    pts  = iou.read_pts(meshname+'.pts')
    elem = iou.read_elem(meshname+'.elem',el_type='Tt',tags=True)

    plt_msh = pts_elem_to_pyvista(pts=pts, elem=elem, add_tags=True)

    if print_fibres:
        lon_initial  = iou.read_lon(meshname+".lon")
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

            visualise_pericardium_all_views(sims_folder=sims_folder,
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
            visualise_epicardium_all_views(sims_folder=sims_folder,
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
            visualise_endocardia_all_views(sims_folder=sims_folder,
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
            visualise_veins_all_views(sims_folder=sims_folder,
                                      plt_msh=plt_msh,
                                      pdf=pdf,
                                      fig_w=fig_w,
                                      fig_h=fig_h,
                                      zoom=zoom,
                                      dpi=dpi,
                                      title_fontsize=title_fontsize,
                                      title_position=title_position)
            
        if print_EAS:
            visualise_EAS_all_views(sims_folder=sims_folder,
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
	   help='Path to the folder containing simulation-ready files.', 
       required=True
    )
    parser.add_argument(
	   '--report_name',
	   type=str,
	   default='output.pdf',
	   help='Output file name.'
    )
    print_options_group = parser.add_argument_group('Print options')
    print_options_group.add_argument(
	   '--fig_w',
	   type=int,
	   default=2480,
	   help='Page width in pixels.'
    )
    print_options_group.add_argument(
	   '--fig_h',
	   type=int,
	   default=3508,
	   help='Page height in pixels.'
    )
    print_options_group.add_argument(
	   '--colormap',
	   type=str,
	   default='RdBu',
	   help='Matplotlib colormap for tags.'
    )
    print_options_group.add_argument(
	   '--zoom',
	   type=float,
	   default=1,
	   help='Zoom magnitude.'
    )
    print_options_group.add_argument(
	   '--dpi',
	   type=float,
	   default=100,
	   help='Dots per inch (resolution).'
    )
    print_options_group.add_argument(
	   '--title_fontsize',
	   type=float,
	   default=44,
	   help='Fontsize of the title of each page.'
    )
    print_options_group.add_argument(
	   '--title_position',
	   type=float,
	   default=0.9,
	   help='Title position value. 1 is at the top of the page, the lower the value, the lower the position of the title'
    )

    outputs_group = parser.add_argument_group('Available Outputs', 'Choose what to include in the report.')
    outputs_group.add_argument(
	   '--print_whole_mesh', '-whole', 
	   action='store_true',
	   help='Include anterior and posterior mesh views (opaque and translucent).'
    )
    outputs_group.add_argument(
	   '--print_fibres', '-fibres', 
	   action='store_true',
	   help='Include mesh with fibres, separated by chamber, from different views.'
    )
    outputs_group.add_argument(
	   '--print_pericardium', '-peri',
	   action='store_true',
	   help='Include mesh with pericardium penalty map from different views.'
    )
    outputs_group.add_argument(
	   '--print_epicardium', '-epi', 
	   action='store_true',
	   help='Include epicardium surface views.'
    )
    outputs_group.add_argument(
	   '--print_endocardia', '-endo',
	   action='store_true',
	   help='Include endocardia surface views.'
    )
    outputs_group.add_argument(
	   '--print_veins', '-veins',
	   action='store_true',
	   help='Include right pulmonary veins and superior vena cava surface views.'
    )
    outputs_group.add_argument(
	   '--print_EAS', '-eas',
	   action='store_true',
	   help='Include early activation sites (sino-atrial node and ventricular fascicles) views.'
    )
    outputs_group.add_argument(
	   '--print_all', '-all',
	   action='store_true',
        help='Include all possible images in the report.'
    )

    args = parser.parse_args()
    main(args)