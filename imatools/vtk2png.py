import os
import argparse

from common import vtktools as vtku
from common import config  

logger = config.configure_logging(log_name=__name__)

def rm_ext(name):
    return os.path.splitext(name)[0]

def get_vtk_files(base_dir):
    vtk_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.vtk'):
                vtk_files.append(os.path.join(root, file))
    return vtk_files

def main(args) : 
    logger.info('Starting vtk2png')
    base_dir = args.base_dir
    output = os.path.join(base_dir, args.output)
    
    grid_size = tuple(args.grid_size)
    window_size = tuple(args.window_size)

    vtk_files = get_vtk_files(base_dir)

    input_data_type = 'polydata' if args.polydata else 'ugrid' 

    if args.mode == 'single':
        logger.info('Rendering vtk files to a single png')
        vtku.render_vtk_to_single_png(vtk_files, output, grid_size, window_size, input_type=input_data_type)
    elif args.mode == 'multi':
        logger.info('Rendering vtk files to multiple pngs')
        vtku.render_vtk_to_png(vtk_files, output, window_size)
    
    logger.info('Finished vtk2png')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print vtks in a folder to one or many pngs')
    parser.add_argument('mode', choices=['single', 'multi'], help='Mode of operation')
    parser.add_argument('--base-dir', help='Input vtk file')
    parser.add_argument('--output', help='Output png file', default='output.png')
    parser.add_argument('--grid-size', nargs=2, type=int, help='Grid size of the output image', default=[1, 1])
    parser.add_argument('--window-size', nargs=2, type=int, help='Window size of the output image', default=[1000, 1000])
    parser.add_argument('--polydata', action='store_true', help='Use polydata instead of structured')
    args = parser.parse_args()
    main(args)