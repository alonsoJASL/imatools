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
        

if __name__ == "__main__":
    mychoices = [
            'show',  # Show the labels in the mesh
            'carto', 
            'bridges', 
            'convert', 
            'vtk42'
            'sizes'
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

    show_group = input_parser.add_argument_group("show")
    
    args = input_parser.parse_args()
    main(args)




