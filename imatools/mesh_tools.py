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


def main(args): 
    mode = args.mode
    if args.help == False and args.input == "":
        logger.error("Error: No input mesh. Set it with the -in flag.")
        return 1

    if mode == "show":
        execute_show(args)
    elif mode == "carto":
        execute_carto(args)
        

if __name__ == "__main__":
    mychoices = [
            'show',  # Show the labels in the mesh
            'carto'
            ]
    #
    input_parser = argparse.ArgumentParser(description="Extracts a single label from a label map image.")
    input_parser.add_argument("mode", choices=mychoices, help="The mode to run the script in.")
    input_parser.add_argument("help", nargs='?', type=bool, default=False, help="Help page specific to each mode")
    input_parser.add_argument("-in", "--input", type=str, default="", help="The input mesh to be processed.")
    input_parser.add_argument("-ifmt", "--input-format", type=str, choices=['vtk', 'carp'], help="The extension of the input mesh.", default="vtk")
    input_parser.add_argument("-scalars", "--scalar-field", type=str, default="scalars", help="The scalar field to be shown.")

    show_group = input_parser.add_argument_group("show")
    
    args = input_parser.parse_args()
    main(args)




