import os
import argparse
import numpy as np

from common import ioutils as iou
from common import itktools as itku
from common import vtktools as vtku
from common import config  

logger = config.configure_logging(log_name=__name__)

def execute_show(args) : 
    msh_path = args.input 
    msh_dir = os.path.dirname(msh_path)
    msh = os.path.basename(msh_path)

    msh_format = args.input_format

    if msh_format == 'vtk':
        print('vtk - no yet ')
    elif msh_format == 'carp':
        msh_path += '.elem' if '.elem' not in msh else ''
        el = iou.read_elem(msh_path, el_type='Tt', tags=True) 
        el_np = np.array(el)
        tags = el_np[:, -1]
        unique_tags = np.unique(tags)
        output_str = '['
        for tag in unique_tags:
            output_str += f'{tag} '
        output_str += ']'
        print(output_str)
            
def execute_carto(args) :
    msh_path = args.input 
    msh_dir = os.path.dirname(msh_path)
    msh = os.path.basename(msh_path)

    output_msh = os.path.join(msh_dir, 'carto_' + msh)

    vtku.convertToCarto(msh_path, args.scalar_field, output_msh)

def execute_bridges(args) :
    label = args.label
    scalar_field = args.scalar_field
    if label is None:
        logger.error("Error: No label provided. Set it with the -l flag.")
        return 1
    
    msh_path = args.input
    dirname = os.path.dirname(msh_path)
    msh_name = os.path.basename(msh_path)
    output_msh = f'bridges_label_{label}_{msh_name}'
    
    vtk_type = 'ugrid'
    msh_u = vtku.readVtk(msh_path, vtk_type)
    msh = vtku.ugrid2polydata(msh_u)

    polydata = vtku.extract_single_label(msh, label)
    polydata = vtku.clean_mesh(polydata)
    vtku.write_vtk(polydata, dirname, f'clean_{output_msh}')

    # Option 1: Graph-based analysis:
    # Assume polydata is the extracted surface (vtkPolyData)
    # bridgeGraphFlag = vtku.detect_bridges_with_graph(polydata)
    bridgeGraphFlag = vtku.detect_bridges_with_graph(polydata)
    # You can add bridgeGraphFlag to the polydata's cell data for visualization:
    polydata.GetCellData().AddArray(bridgeGraphFlag)

    neighbourhoodFlag = vtku.compute_cell_neighbor_count(polydata)
    polydata.GetCellData().AddArray(neighbourhoodFlag)
    
    # Option 2: Thickness analysis:
    # Choose a max_distance (e.g., based on the scale of your data) and a thickness threshold.
    max_distance = args.max_distance            # Adjust as needed
    thickness_threshold = args.threshold        # Adjust as needed
    bridgeThicknessFlag = vtku.detect_bridges_with_thickness(polydata, max_distance, thickness_threshold, True)
    polydata.GetCellData().AddArray(bridgeThicknessFlag)

    # Save the polydata with the bridge flags:
    vtku.writeVtk(polydata, dirname, output_msh)

def execute_sizes(args) :
    msh_path = args.input 
    msh_dir = os.path.dirname(msh_path)
    base_msh_name = os.path.basename(msh_path)
    base_msh_name = base_msh_name.split('.')[0]

    # find names of the type f'{base_msh_name}.part{n}.vtk' 
    parts = []
    for file in os.listdir(msh_dir):
        if file.endswith(".vtk") and file.startswith(base_msh_name):
            parts.append(os.path.join(msh_dir, file))
    
    # read the parts and compute the size of each one
    answers = []
    for fi in parts : 
        poly = vtku.read_vtk(fi, 'ugrid')
        num_cells, total_area = vtku.compute_mesh_size(poly)
        answers.append(f'{os.path.basename(fi)} has {num_cells} cells and total area {total_area}')
    
    for ans in answers:
        print(ans)


def execute_convert(args) :
    msh_path = args.input

    if args.input_format == 'stl' : 
        msh_clean_path = os.path.join(os.path.dirname(msh_path), 'clean_' + os.path.basename(msh_path))
        vtku.clean_stl_file(msh_path, msh_clean_path)
        msh_path = msh_clean_path

    msh_dir = os.path.dirname(msh_path)
    msh = os.path.basename(msh_path).split('.')[0]

    output_msh = os.path.join(msh_dir, f'{msh}.{args.output_format}')

    loaded_mesh = vtku.read_vtk(msh_path, input_type=args.input_format)
    vtku.export_as(loaded_mesh, output_msh, export_as=args.output_format)

def execute_shell_to_image(args) :
    msh_path = args.input
    msh_dir = os.path.dirname(msh_path)
    msh = os.path.basename(msh_path)
    output_img = msh.replace('.vtk', '.nrrd')

    output_img = os.path.join(msh_dir, f'{output_img}')

    ref_img = args.reference_image 
    if ref_img is None:
        logger.error("Error: No reference image provided. Set it with the -ref flag.")
        return 1
    
    loaded_mesh = vtku.read_vtk(msh_path)
    ref_img = itku.load_image(ref_img)

    if args.mesh_list_folder is not None:
        mesh_list = []
        mesh_files_list_folder = os.listdir(args.mesh_list_folder)
        for mesh_file in mesh_files_list_folder:
            if mesh_file.endswith('.vtk'):
                msh_name = os.path.join(args.mesh_list_folder, mesh_file)
                mesh_list.append(vtku.read_vtk(msh_name)) 
            
        combined_bounds = vtku.get_combined_bounds(mesh_list)
        ref_img = vtku.create_image_with_combined_origin(ref_img, combined_bounds)            
    
    shell_img = vtku.mesh_to_image(loaded_mesh, ref_img, inside_value=1, outside_value=0, reverse_stencil=args.reverse_stencil)
    cc_image, labels, num_labels = itku.bwlabeln(shell_img)

    logger.info(f"Number of labels: {num_labels}")

    itku.save_image(cc_image, output_img)

def execute_scar3d(args) :
    msh_path = args.input
    msh_dir = os.path.dirname(msh_path)
    msh_name = os.path.basename(msh_path)
    output_msh_name = os.path.join(msh_dir, f'scar3d_{msh_name}')

    img = itku.load_image(args.reference_image)
    label = args.label

    vox_indices, _, bboxes_dict = itku.get_indices_from_label(img, label, get_voxel_bbox=True)
    bboxes = bboxes_dict['corners']
    logger.info(f'Found {len(vox_indices)} voxels for label {label}, and {len(bboxes)} bounding boxes.')

    msh = vtku.read_vtk(msh_path, input_type='ugrid')

    outmsh = vtku.tag_elements_by_voxel_boxes(msh, bboxes, label_name='scar')
    vtku.write_vtk(outmsh, msh_dir, output_msh_name, output_type='ugrid')

def execute_scar_w_cog(args) : 
    msh_path = args.input 
    msh_dir = os.path.dirname(msh_path)
    msh_name = os.path.basename(msh_path) 
    cog_path = os.path.join(msh_dir, msh_name.replace('.vtk', '.pts'))
    output_msh_name = f'scar3d_{msh_name}'

    cogs = np.loadtxt(cog_path)

    img = itku.load_image(args.reference_image)
    label = args.label 

    vox_indices, _, bboxes_dict = itku.get_indices_from_label(img, label, get_voxel_bbox=True)
    bboxes = bboxes_dict['corners']
    logger.info(f'Found {len(vox_indices)} voxels with label {args.label}')

    msh = vtku.read_vtk(msh_path, input_type='ugrid') 

    outmsh = vtku.tag_mesh_elements_by_voxel_boxes(msh, cogs, bboxes)
    vtku.write_vtk(outmsh, msh_dir, output_msh_name, output_type='ugrid') 

def execute_smart_scar(args) :
    msh_path = args.input 
    msh_dir = os.path.dirname(msh_path)
    msh_name = os.path.basename(msh_path) 
    cog_path = os.path.join(msh_dir, msh_name.replace('.vtk', '.pts'))
    output_msh_name = f'scar3d_{msh_name}'

    cogs = np.loadtxt(cog_path)

    img = itku.load_image(args.reference_image)
    label = args.label 

    vox_indices, _, bboxes_dict = itku.get_indices_from_label(img, label, get_voxel_bbox=True)
    label = args.label

    msh= vtku.read_vtk(msh_path, input_type='ugrid')

    outmsh = vtku.tag_mesh_elements_by_growing_from_seed(msh, bboxes_dict['centres'], bboxes_dict['corners'], cogs=cogs, label_name='scar')
    vtku.write_vtk(outmsh, msh_dir, output_msh_name, output_type='ugrid')

def execute_bbox_compare(args) :
    msh_path = args.input
    msh_dir = os.path.dirname(msh_path)
    msh_name = os.path.basename(msh_path)

    img = itku.load_image(args.reference_image)
    label = args.label
    if label < 1 or label is None: # binarise the whole image and use label=1
        img = itku.binarise(img)
        label = 1

    vox_indices, world_coords = itku.get_indices_from_label(img, label, get_voxel_bbox=False)
    logger.info(f'Found {len(vox_indices)} voxels for label {label}.')

    # get total bounding box from world coordinates
    min_coords = np.min(world_coords, axis=0)
    max_coords = np.max(world_coords, axis=0)

    bbox_img = np.array([min_coords, max_coords])

    msh = vtku.read_vtk(msh_path, input_type='ugrid')
    bbox_msh = vtku.get_bounding_box(msh)
    
    logger.info(f'Bounding box from mesh: {bbox_msh}')
    logger.info(f'Bounding box from image: {bbox_img}')

def execute_cog(args) : 
    msh_path = args.input 
    msh_dir = os.path.dirname(msh_path)
    msh_name = os.path.basename(msh_path) 

    output_name = msh_name.replace('.vtk', '.pts')
    msh = vtku.read_vtk(msh_path, input_type='ugrid') 
    cogs = vtku.cogs_from_ugrid(msh) 

    logger.info('Saving file...')
    np.savetxt(os.path.join(msh_dir, output_name), cogs, delimiter=' ') 


def execute_bbox(args) : 
    input_path = args.input 
    if input_path.endswith('.vtk') : 
        msh = vtku.read_vtk(input_path, input_type='ugrid')
        bbox = vtku.get_bounding_box(msh) 
    else : 
        img = itku.load_image(input_path) 
        img = itku.binarise(img) 
        _, world_coords = itku.get_indices_from_label(img, 1, get_voxel_bbox=False) 

        min_coords = np.min(world_coords, axis=0)
        max_coords = np.max(world_coords, axis=0)
        bbox = np.array([min_coords, max_coords])

    logger.info(f'{bbox=}')


def main(args): 
    mode = args.mode
    if args.help == False and args.input == "":
        logger.error("Error: No input mesh. Set it with the -in flag.")
        return 1

    if mode == "show":
        execute_show(args)
    elif mode == "carto":
        execute_carto(args)
    elif mode == "bridges":
        execute_bridges(args)
    elif mode == "convert":
        execute_convert(args)
    elif mode == "sizes":
        execute_sizes(args)
    elif mode == "shell_to_image":
        execute_shell_to_image(args)
    elif mode == "cog":
        execute_cog(args)
    elif mode == "scar3d":
        execute_scar3d(args)
    elif mode =='cogscar3d' : 
        execute_scar_w_cog(args)
    elif mode == 'smart_scar' :
        execute_smart_scar(args)
    elif mode == 'bbox_compare':
        execute_bbox_compare(args)
    elif mode == 'bbox' : 
        execute_bbox(args)
    else:
        logger.error(f"Unknown mode: {mode}. Please choose from {', '.join(mychoices)}")
        return 1

if __name__ == "__main__":
    mychoices = [
            'show',  # Show the labels in the mesh
            'carto', 
            'bridges', 
            'convert', 
            'vtk42',
            'sizes',
            'shell_to_image', 
            'cog', 
            'scar3d', 
            'cogscar3d',
            'smart_scar',
            'bbox_compare', 
            'bbox'
            ]
    #
    input_parser = argparse.ArgumentParser(description="Extracts a single label from a label map image.")
    input_parser.add_argument("mode", choices=mychoices, help="The mode to run the script in.")
    input_parser.add_argument("help", nargs='?', type=bool, default=False, help="Help page specific to each mode")
    input_parser.add_argument("-in", "--input", type=str, default="", help="The input mesh to be processed.")
    input_parser.add_argument("-ifmt", "--input-format", type=str, choices=['vtk', 'carp','stl'], help="The extension of the input mesh.", default="vtk")
    input_parser.add_argument("-ofmt", "--output-format", type=str, choices=vtku.EXPORT_DATA_TYPES, help="The output mesh to be saved.")
    input_parser.add_argument("-scalars", "--scalar-field", type=str, default="scalars", help="The scalar field to be shown.")

    bridges_group = input_parser.add_argument_group("bridges")
    bridges_group.add_argument('-l', '--label', type=int, default=None, help='The label to extract')
    bridges_group.add_argument('-max-distance', '--max-distance', type=float, default=5.0, help='The maximum distance to search for bridges')
    bridges_group.add_argument('-threshold', '--threshold', type=float, default=1.5, help='The thickness threshold to detect bridges')

    sh2im_group = input_parser.add_argument_group("Shell to image")
    sh2im_group.add_argument('-ref', '--reference-image', type=str, default=None, help='The reference image to map the shell to')
    sh2im_group.add_argument('-rs', '--reverse-stencil', action='store_true', help='Reverse the stencil')
    sh2im_group.add_argument('-mesh-list-folder', '--mesh-list-folder', type=str, default=None, help='List of meshes to process')

    
    args = input_parser.parse_args()
    main(args)




